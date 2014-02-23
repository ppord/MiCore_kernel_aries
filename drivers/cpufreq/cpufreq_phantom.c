/*
 *  drivers/cpufreq/cpufreq_phantom.c
 *
 *  Copyright (C)  2001 Russell King
 *            (C)  2003 Venkatesh Pallipadi <venkatesh.pallipadi@intel.com>.
 *                      Jun Nakajima <jun.nakajima@intel.com>
 *            (C)  2009 Alexander Clouter <alex@digriz.org.uk>
 *	      (c)  2014 Jake van der Putten <jakevdputten@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/cpufreq.h>
#include <linux/cpu.h>
#include <linux/jiffies.h>
#include <linux/kernel_stat.h>
#include <linux/mutex.h>
#include <linux/hrtimer.h>
#include <linux/tick.h>
#include <linux/ktime.h>
#include <linux/sched.h>
#include <linux/input.h>
#include <linux/slab.h>
#include <linux/earlysuspend.h>

/*
 * dbs is used in this file as a shortform for demandbased switching
 * It helps to keep variable names smaller, simpler
 */

#define DEF_MIN_SAMPLE_TIME			(4000)
#define DEF_UP_THRESHOLD			(90)
#define DEF_DOWN_THRESHOLD			(20)
#define MIN_FREQUENCY_THRESHOLD			(11)
#define MAX_FREQUENCY_THRESHOLD			(100)
#define DEF_FREQ_STEP				(5)
#define DEF_INPUT_BOOST				(1)
#define DEF_INPUT_BOOST_TIME			(4000)
#define DEF_IDEAL_FREQUENCY			(1242000)
#define DEF_INPUT_FREQUENCY			(1242000)
#define DEF_SUSPEND_MAX_FREQ			(918000)

/*
 * Smart scaling increases the cpu frequency by calculating freq_smart.
 * freq_smart is calculated by calculating freq_avg (policy->min/max + policy->cur / 2)
 * freq_smart is freq_avg * smart_freq_increase (in percent)
 */
#define DEF_SMART_SCALING			(1)
#define DEF_SMART_POWERSAVE			(0)
#define DEF_SMART_FREQ_INCREASE			(0)

/*
 * Phantom mode defines the default behaviour of the phantom governor
 * when the device is not suspended. There are in total 4 profiles supported.
 * (0) conservative governor
 * (1) phantom governor (powersave)(smart scaling)
 * (2) phantom governor (default)(smart scaling)
 * (3) phantom governor (static)(ideal scaling)
 */
#define DEF_PHANTOM_MODE			(2)

/* Variables for phantom specific features */
static unsigned int suspended; 
unsigned int input = false; 
unsigned int phantom_mode_update = true;
u64 last_freq_increase;
u64 last_input_time;

/*
 * The polling frequency of this governor depends on the capability of
 * the processor. Default polling frequency is 1000 times the transition
 * latency of the processor. The governor will work on any processor with
 * transition latency <= 10mS, using appropriate sampling
 * rate.
 * For CPUs with transition latency > 10mS (mostly drivers with CPUFREQ_ETERNAL)
 * this governor will not work.
 * All times here are in uS.
 */
#define MIN_SAMPLING_RATE_RATIO			(2)

static unsigned int min_sampling_rate;

#define LATENCY_MULTIPLIER			(1000)
#define MIN_LATENCY_MULTIPLIER			(100)
#define TRANSITION_LATENCY_LIMIT		(10 * 1000 * 1000)

static void do_dbs_timer(struct work_struct *work);

struct cpu_dbs_info_s {
	cputime64_t prev_cpu_idle;
	cputime64_t prev_cpu_wall;
	cputime64_t prev_cpu_nice;
	struct cpufreq_frequency_table *freq_table;
	struct cpufreq_policy *cur_policy;
	struct delayed_work work;
	unsigned int down_skip;
	unsigned int requested_freq;
	int cpu;
	unsigned int enable:1;
	/*
	 * percpu mutex that serializes governor limit change with
	 * do_dbs_timer invocation. We do not want do_dbs_timer to run
	 * when user is changing the governor or limits.
	 */
	struct mutex timer_mutex;
};
static DEFINE_PER_CPU(struct cpu_dbs_info_s, cs_cpu_dbs_info);

static unsigned int dbs_enable;	/* number of CPUs using this policy */

/*
 * dbs_mutex protects dbs_enable in governor start/stop.
 */
static DEFINE_MUTEX(dbs_mutex);

static struct workqueue_struct *dbs_wq;

static struct dbs_tuners {
	unsigned int sampling_rate;
	unsigned int min_sample_time;
	unsigned int up_threshold;
	unsigned int down_threshold;
	unsigned int phantom_mode;
	unsigned int freq_step;
	unsigned int ideal_freq;
	unsigned int input_boost;
	unsigned int input_boost_freq;
	unsigned int input_boost_time;
	unsigned int smart_scaling;
	unsigned int smart_powersave;
	unsigned int smart_freq_increase;
	unsigned int suspend_max_freq;
} dbs_tuners_ins = {
	.up_threshold = DEF_UP_THRESHOLD,
	.down_threshold = DEF_DOWN_THRESHOLD,
	.min_sample_time = DEF_MIN_SAMPLE_TIME,
	.phantom_mode = DEF_PHANTOM_MODE,
	.freq_step = DEF_FREQ_STEP,
	.ideal_freq = DEF_IDEAL_FREQUENCY,
	.input_boost = DEF_INPUT_BOOST,
	.input_boost_freq = DEF_INPUT_FREQUENCY,
	.input_boost_time = DEF_INPUT_BOOST_TIME,
	.smart_scaling = DEF_SMART_SCALING,
	.smart_powersave = DEF_SMART_POWERSAVE,
	.smart_freq_increase = DEF_SMART_FREQ_INCREASE,
	.suspend_max_freq = DEF_SUSPEND_MAX_FREQ,
};

static inline u64 get_cpu_idle_time_jiffy(unsigned int cpu, u64 *wall)
{
	u64 idle_time;
	u64 cur_wall_time;
	u64 busy_time;

	cur_wall_time = jiffies64_to_cputime64(get_jiffies_64());

	busy_time  = kcpustat_cpu(cpu).cpustat[CPUTIME_USER];
	busy_time += kcpustat_cpu(cpu).cpustat[CPUTIME_SYSTEM];
	busy_time += kcpustat_cpu(cpu).cpustat[CPUTIME_IRQ];
	busy_time += kcpustat_cpu(cpu).cpustat[CPUTIME_SOFTIRQ];
	busy_time += kcpustat_cpu(cpu).cpustat[CPUTIME_STEAL];
	busy_time += kcpustat_cpu(cpu).cpustat[CPUTIME_NICE];

	idle_time = cur_wall_time - busy_time;
	if (wall)
		*wall = jiffies_to_usecs(cur_wall_time);

	return jiffies_to_usecs(idle_time);
}

static inline cputime64_t get_cpu_idle_time(unsigned int cpu, cputime64_t *wall)
{
	u64 idle_time = get_cpu_idle_time_us(cpu, NULL);

	if (idle_time == -1ULL)
		return get_cpu_idle_time_jiffy(cpu, wall);
	else
		idle_time += get_cpu_iowait_time_us(cpu, wall);

	return idle_time;
}

/* keep track of frequency transitions */
static int
dbs_cpufreq_notifier(struct notifier_block *nb, unsigned long val,
		     void *data)
{
	struct cpufreq_freqs *freq = data;
	struct cpu_dbs_info_s *this_dbs_info = &per_cpu(cs_cpu_dbs_info,
							freq->cpu);

	struct cpufreq_policy *policy;

	if (!this_dbs_info->enable)
		return 0;

	policy = this_dbs_info->cur_policy;

	/*
	 * we only care if our internally tracked freq moves outside
	 * the 'valid' ranges of freqency available to us otherwise
	 * we do not change it
	*/
	if (this_dbs_info->requested_freq > policy->max
			|| this_dbs_info->requested_freq < policy->min)
		this_dbs_info->requested_freq = freq->new;

	return 0;
}

static struct notifier_block dbs_cpufreq_notifier_block = {
	.notifier_call = dbs_cpufreq_notifier
};

/************************** sysfs interface ************************/
static ssize_t show_sampling_rate_min(struct kobject *kobj,
				      struct attribute *attr, char *buf)
{
	return sprintf(buf, "%u\n", min_sampling_rate);
}

define_one_global_ro(sampling_rate_min);

/* cpufreq_phantom Governor Tunables */
#define show_one(file_name, object)					\
static ssize_t show_##file_name						\
(struct kobject *kobj, struct attribute *attr, char *buf)		\
{									\
	return sprintf(buf, "%u\n", dbs_tuners_ins.object);		\
}
show_one(sampling_rate, sampling_rate);
show_one(min_sample_time, min_sample_time);
show_one(up_threshold, up_threshold);
show_one(down_threshold, down_threshold);
show_one(phantom_mode, phantom_mode);
show_one(freq_step, freq_step);
show_one(ideal_freq, ideal_freq);
show_one(input_boost, input_boost);
show_one(input_boost_freq, input_boost_freq);
show_one(input_boost_time, input_boost_time);
show_one(smart_scaling, smart_scaling);
show_one(smart_powersave, smart_powersave);
show_one(smart_freq_increase, smart_freq_increase);
show_one(suspend_max_freq, suspend_max_freq);

/**
 * update_sampling_rate - update sampling rate effective immediately if needed.
 * @new_rate: new sampling rate
 *
 * If new rate is smaller than the old, simply updaing
 * dbs_tuners_int.sampling_rate might not be appropriate. For example,
 * if the original sampling_rate was 1 second and the requested new sampling
 * rate is 10 ms because the user needs immediate reaction from ondemand
 * governor, but not sure if higher frequency will be required or not,
 * then, the governor may change the sampling rate too late; up to 1 second
 * later. Thus, if we are reducing the sampling rate, we need to make the
 * new value effective immediately.
 */
static void update_sampling_rate(unsigned int new_rate)
{
	int cpu;

	dbs_tuners_ins.sampling_rate = new_rate
				     = max(new_rate, min_sampling_rate);

	get_online_cpus();
	for_each_online_cpu(cpu) {
		struct cpufreq_policy *policy;
		struct cpu_dbs_info_s *dbs_info;
		unsigned long next_sampling, appointed_at;

		policy = cpufreq_cpu_get(cpu);
		if (!policy)
			continue;
		dbs_info = &per_cpu(cs_cpu_dbs_info, policy->cpu);
		cpufreq_cpu_put(policy);

		mutex_lock(&dbs_info->timer_mutex);

		if (!delayed_work_pending(&dbs_info->work)) {
			mutex_unlock(&dbs_info->timer_mutex);
			continue;
		}

		next_sampling  = jiffies + usecs_to_jiffies(new_rate);
		appointed_at = dbs_info->work.timer.expires;


		if (time_before(next_sampling, appointed_at)) {

			mutex_unlock(&dbs_info->timer_mutex);
			cancel_delayed_work_sync(&dbs_info->work);
			mutex_lock(&dbs_info->timer_mutex);

			queue_delayed_work_on(dbs_info->cpu, dbs_wq,
				&dbs_info->work, usecs_to_jiffies(new_rate));

		}
		mutex_unlock(&dbs_info->timer_mutex);
	}
	put_online_cpus();
}

static ssize_t store_sampling_rate(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	update_sampling_rate(input);
	return count;
}

static ssize_t store_min_sample_time(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;

	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.min_sample_time = input;
	return count;
}

static ssize_t store_ideal_freq(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;

	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.ideal_freq = input;
	return count;
}

static ssize_t store_input_boost(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.input_boost = input;
	return count;
}

static ssize_t store_input_boost_freq(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;

	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.input_boost_freq = input;
	return count;
}

static ssize_t store_input_boost_time(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.input_boost_time = input;
	return count;
}

static ssize_t store_smart_scaling(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.smart_scaling = input;
	return count;
}

static ssize_t store_smart_powersave(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.smart_powersave = input;
	return count;
}

static ssize_t store_smart_freq_increase(struct kobject *a, struct attribute *b,
				    const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.smart_freq_increase = input;
	return count;
}

static ssize_t store_suspend_max_freq(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;

	ret = sscanf(buf, "%u", &input);
	if (ret != 1)
		return -EINVAL;
	dbs_tuners_ins.suspend_max_freq = input;
	return count;
}

static ssize_t store_up_threshold(struct kobject *a, struct attribute *b,
				  const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);

	if (ret != 1 || input > MAX_FREQUENCY_THRESHOLD ||
			input < MIN_FREQUENCY_THRESHOLD) {
		return -EINVAL;
	}
	dbs_tuners_ins.up_threshold = input;
	return count;
}

static ssize_t store_down_threshold(struct kobject *a, struct attribute *b,
				    const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);

	if (ret != 1 || input > MAX_FREQUENCY_THRESHOLD ||
			input < MIN_FREQUENCY_THRESHOLD) {
		return -EINVAL;
	}
	dbs_tuners_ins.down_threshold = input;
	return count;
}

static ssize_t store_phantom_mode(struct kobject *a, struct attribute *b,
				   const char *buf, size_t count)
{
	unsigned int input;
	int ret;

	ret = sscanf(buf, "%u", &input);
	if (ret != 1 || input > 3)
		return -EINVAL;

	dbs_tuners_ins.phantom_mode = input;
	phantom_mode_update = true;

	return count;
}

static ssize_t store_freq_step(struct kobject *a,
			struct attribute *b, const char *buf, size_t count)
{
	unsigned int input;
	int ret;
	ret = sscanf(buf, "%u", &input);

	if (ret != 1 || input > 100 ||
			input < 0) {
		return -EINVAL;
	}
	dbs_tuners_ins.freq_step = input;
	return count;
}

define_one_global_rw(sampling_rate);
define_one_global_rw(min_sample_time);
define_one_global_rw(up_threshold);
define_one_global_rw(down_threshold);
define_one_global_rw(phantom_mode);
define_one_global_rw(freq_step);
define_one_global_rw(ideal_freq);
define_one_global_rw(input_boost);
define_one_global_rw(input_boost_freq);
define_one_global_rw(input_boost_time);
define_one_global_rw(smart_scaling);
define_one_global_rw(smart_powersave);
define_one_global_rw(smart_freq_increase);
define_one_global_rw(suspend_max_freq);

static struct attribute *dbs_attributes[] = {
	&sampling_rate_min.attr,
	&sampling_rate.attr,
	&min_sample_time.attr,
	&up_threshold.attr,
	&down_threshold.attr,
	&phantom_mode.attr,
	&freq_step.attr,
	&ideal_freq.attr,
	&input_boost.attr,
	&input_boost_freq.attr,
	&input_boost_time.attr,
	&smart_scaling.attr,
	&smart_powersave.attr,
	&smart_freq_increase.attr,
	&suspend_max_freq.attr,
	NULL
};

static struct attribute_group dbs_attr_group = {
	.attrs = dbs_attributes,
	.name = "phantom",
};

/************************** Phantom parts *******************/
/* Generic function to update frequency */
static void phantom_update_freq(struct cpufreq_policy *p, unsigned int new_freq)
{
__cpufreq_driver_target(p, new_freq, CPUFREQ_RELATION_H);
}

/* Increase frequency by one step using frequency table */
static unsigned int phantom_freq_increase(struct cpu_dbs_info_s *this_dbs_info, 
						unsigned int current_freq)
{
	unsigned int index = 0;
	unsigned int target_freq;
	struct cpufreq_policy *policy;
	
	policy = this_dbs_info->cur_policy;

	cpufreq_frequency_table_target(policy, 
					this_dbs_info->freq_table, 
					current_freq + 1, 
					CPUFREQ_RELATION_L, &index);
	target_freq = this_dbs_info->freq_table[index].frequency;

	return target_freq;
}

/* Decrease frequency by one step using frequency table */
static unsigned int phantom_freq_decrease(struct cpu_dbs_info_s *this_dbs_info, 
						unsigned int current_freq)
{
	unsigned int index = 0;
	unsigned int target_freq;
	struct cpufreq_policy *policy;
	
	policy = this_dbs_info->cur_policy;

	cpufreq_frequency_table_target(policy, 
					this_dbs_info->freq_table, 
					current_freq - 1, 
					CPUFREQ_RELATION_H, &index);
	target_freq = this_dbs_info->freq_table[index].frequency;

	return target_freq;
}

/* 
 * Use frequency table to find frequency closest to smart_freq 
 * If energy efficient smart scaling is enabled smart_freq will be rounded down
 * else it will be rounded up.
 */
static unsigned int phantom_freq_smart(struct cpu_dbs_info_s *this_dbs_info, 
						unsigned int current_freq,
						unsigned int relation)
{
	unsigned int index = 0;
	unsigned int target_freq;
	struct cpufreq_policy *policy;
	
	policy = this_dbs_info->cur_policy;

	cpufreq_frequency_table_target(policy, 
					this_dbs_info->freq_table, 
					current_freq, relation, &index);
	target_freq = this_dbs_info->freq_table[index].frequency;

	return target_freq;
}

static void phantom_update_mode(void)
{
	phantom_mode_update = false;
	switch (dbs_tuners_ins.phantom_mode) {
	case 0:
		dbs_tuners_ins.down_threshold = 20;
		dbs_tuners_ins.up_threshold = 80;
		dbs_tuners_ins.freq_step = 5;
		break;

	case 1:
		dbs_tuners_ins.up_threshold = 95;
		dbs_tuners_ins.down_threshold = 25;
		dbs_tuners_ins.min_sample_time = 2000;
		dbs_tuners_ins.freq_step = DEF_FREQ_STEP;
		dbs_tuners_ins.ideal_freq = DEF_IDEAL_FREQUENCY;
		dbs_tuners_ins.input_boost = DEF_INPUT_BOOST;
		dbs_tuners_ins.input_boost_freq = DEF_INPUT_FREQUENCY;
		dbs_tuners_ins.input_boost_time = DEF_INPUT_BOOST_TIME;
		dbs_tuners_ins.smart_scaling = DEF_SMART_SCALING;
		dbs_tuners_ins.smart_powersave = 1;
		dbs_tuners_ins.smart_freq_increase = DEF_SMART_FREQ_INCREASE;
		break;

	case 2:
		dbs_tuners_ins.up_threshold = DEF_UP_THRESHOLD;
		dbs_tuners_ins.down_threshold = DEF_DOWN_THRESHOLD;
		dbs_tuners_ins.min_sample_time = DEF_MIN_SAMPLE_TIME;
		dbs_tuners_ins.freq_step = DEF_FREQ_STEP;
		dbs_tuners_ins.ideal_freq = DEF_IDEAL_FREQUENCY;
		dbs_tuners_ins.input_boost = DEF_INPUT_BOOST;
		dbs_tuners_ins.input_boost_freq = DEF_INPUT_FREQUENCY;
		dbs_tuners_ins.input_boost_time = DEF_INPUT_BOOST_TIME;
		dbs_tuners_ins.smart_scaling = DEF_SMART_SCALING;
		dbs_tuners_ins.smart_powersave = DEF_SMART_POWERSAVE;
		dbs_tuners_ins.smart_freq_increase = DEF_SMART_FREQ_INCREASE;
		break;

	case 3:
		dbs_tuners_ins.up_threshold = DEF_UP_THRESHOLD;
		dbs_tuners_ins.down_threshold = DEF_DOWN_THRESHOLD;
		dbs_tuners_ins.min_sample_time = 6000;
		dbs_tuners_ins.freq_step = DEF_FREQ_STEP;
		dbs_tuners_ins.ideal_freq = DEF_IDEAL_FREQUENCY;
		dbs_tuners_ins.input_boost = DEF_INPUT_BOOST;
		dbs_tuners_ins.input_boost_freq  = DEF_INPUT_FREQUENCY;
		dbs_tuners_ins.input_boost_time = 2000;
		dbs_tuners_ins.smart_scaling = 0;
		dbs_tuners_ins.smart_powersave = DEF_SMART_POWERSAVE;
		dbs_tuners_ins.smart_freq_increase = DEF_SMART_FREQ_INCREASE;
		break;

	}
}	

/************************ Phantom parts end *******************/	
/*
 * Input boost interface: when smart scaling is enabled input_boost will use 
 * freq_smart, else it will use predefined input_boost_freq
 */
static void phantom_input_boost(struct cpu_dbs_info_s *this_dbs_info)
{
	unsigned int freq_target = 0;
	struct cpufreq_policy *policy;

	policy = this_dbs_info->cur_policy;

	if (input) {
		input = false;

		if (dbs_tuners_ins.input_boost && dbs_tuners_ins.phantom_mode) {
			if (policy->cur < dbs_tuners_ins.input_boost_freq) {
				freq_target = dbs_tuners_ins.input_boost_freq;
				last_freq_increase = ktime_to_ms(ktime_get());
				phantom_update_freq(policy, freq_target);
				return;
			}
		}	
	}
}

/* Phantom governor: used when not suspended and used when phantom mode is higer then 0 */
static void dbs_check_cpu(struct cpu_dbs_info_s *this_dbs_info)
{
	if (!suspended || dbs_tuners_ins.phantom_mode) {
		u64 now;
		unsigned int load = 0;
		unsigned int max_load = 0;
		unsigned int max_load_any_cpu = 0;
		unsigned int freq_target;

		struct cpufreq_policy *policy;
		unsigned int j;

		policy = this_dbs_info->cur_policy;

		if (phantom_mode_update)
			phantom_update_mode();

		/*
		 * Every sampling_rate, we check, if current idle time is less
		 * than 20% (default), then we try to increase frequency
		 * Every sampling_rate*sampling_down_factor, we check, if current
		 * idle time is more than 80%, then we try to decrease frequency
		 *
		 * Any frequency increase takes it to the maximum frequency.
		 * Frequency reduction happens at minimum steps of
		 * 5% (default) of maximum frequency
		 */

		/* Get Absolute Load */
		for_each_cpu(j, policy->cpus) {
			struct cpu_dbs_info_s *j_dbs_info;
			cputime64_t cur_wall_time, cur_idle_time;
			unsigned int idle_time, wall_time;

			j_dbs_info = &per_cpu(cs_cpu_dbs_info, j);
	
			cur_idle_time = get_cpu_idle_time(j, &cur_wall_time);

			wall_time = (unsigned int)
				(cur_wall_time - j_dbs_info->prev_cpu_wall);
			j_dbs_info->prev_cpu_wall = cur_wall_time;

			idle_time = (unsigned int)
				(cur_idle_time - j_dbs_info->prev_cpu_idle);
			j_dbs_info->prev_cpu_idle = cur_idle_time;

			if (unlikely(!wall_time || wall_time < idle_time))
				continue;
	
			load = 100 * (wall_time - idle_time) / wall_time;

			if (load > max_load)
				max_load = load;

			if (load > max_load_any_cpu)
				max_load_any_cpu = load;
		}

		/* Check for frequency increase */
		if (max_load > dbs_tuners_ins.up_threshold) {
			this_dbs_info->down_skip = 0;

			/* If we are already at full speed then break out early */
			if (policy->cur == policy->max)
				return;

			/*
			 * If smart scaling is enabled calculate freq_avg and freq_smart, 
			 * else use predefined ideal_freq to scale up frequency
  			 */
			if (dbs_tuners_ins.smart_scaling) {
				unsigned int freq_avg;
				unsigned int smart_policy;
				unsigned int freq_smart;

				freq_avg = (policy->cur + policy->max) / 2;
				smart_policy = (freq_avg * (100 + dbs_tuners_ins.smart_freq_increase)) / 100;
				if (dbs_tuners_ins.smart_powersave)
					freq_smart = phantom_freq_smart(this_dbs_info, smart_policy, CPUFREQ_RELATION_H);
				else
					freq_smart = phantom_freq_smart(this_dbs_info, smart_policy, CPUFREQ_RELATION_L);

				if (policy->cur < freq_smart) 
					freq_target = freq_smart;
				else 
					freq_target = phantom_freq_increase(this_dbs_info, policy->cur);
				
			} else {
				if (policy->cur < dbs_tuners_ins.ideal_freq) {
					freq_target = dbs_tuners_ins.ideal_freq;
				} else {
					freq_target = phantom_freq_increase(this_dbs_info, policy->cur);
				}
			}

			if (freq_target > policy->max)
				freq_target = policy->max;

			if (freq_target)
				last_freq_increase = ktime_to_ms(ktime_get());
				phantom_update_freq(policy, freq_target);
				return;
		}

		/*
		 * If current time - last_freq_increase is lower then
		 * min_sample_time break out early.
		 */
		now = ktime_to_ms(ktime_get());
		if (now - last_freq_increase < dbs_tuners_ins.min_sample_time)
			return;

		/*
		 * The optimal frequency is the frequency that is the lowest that
		 * can support the current CPU usage without triggering the up
		 * policy. To be safe, we focus 10 points under the threshold.
		 */
		if (max_load < (dbs_tuners_ins.down_threshold - 10)) {

			/* If we cannot reduce the frequency anymore, break out early */
			if (policy->cur == policy->min)
				return;
	
			if (dbs_tuners_ins.smart_scaling) {
				unsigned int freq_avg;
				unsigned int smart_policy;
				unsigned int freq_smart;

				freq_avg = (policy->cur + policy->min) / 2;
				smart_policy = (freq_avg * (100 + dbs_tuners_ins.smart_freq_increase)) / 100;
				freq_smart = phantom_freq_smart(this_dbs_info, smart_policy, CPUFREQ_RELATION_H);

				if (policy->cur > freq_smart) {
					freq_target = freq_smart;
				} else {
					freq_target = phantom_freq_decrease(this_dbs_info, policy->cur);
				}
			} else {
				freq_target = phantom_freq_decrease(this_dbs_info, policy->cur);
			}

			if (freq_target < policy->min)
				freq_target = policy->min;

			last_freq_increase = 0;

			if (freq_target)
				phantom_update_freq(policy, freq_target);
				return;
		}
	}
}

/* Conservative governor: used when suspended and when phantom_mode is 0 */
static void dbs_check_cpu_suspend(struct cpu_dbs_info_s *this_dbs_info)
{
	if (suspended || !dbs_tuners_ins.phantom_mode) {
		unsigned int load = 0;
		unsigned int max_load = 0;
		unsigned int freq_target;

		struct cpufreq_policy *policy;
		unsigned int j;
	
		policy = this_dbs_info->cur_policy;

		if (phantom_mode_update)
			phantom_update_mode();

		/*
		 * Every sampling_rate, we check, if current idle time is less
		 * than 20% (default), then we try to increase frequency
		 * Every sampling_rate*sampling_down_factor, we check, if current
		 * idle time is more than 80%, then we try to decrease frequency
		 *
		 * Any frequency increase takes it to the maximum frequency.
		 * Frequency reduction happens at minimum steps of
		 * 5% (default) of maximum frequency
		 */

		/* Get Absolute Load */
		for_each_cpu(j, policy->cpus) {
			struct cpu_dbs_info_s *j_dbs_info;
			cputime64_t cur_wall_time, cur_idle_time;
			unsigned int idle_time, wall_time;

			j_dbs_info = &per_cpu(cs_cpu_dbs_info, j);

			cur_idle_time = get_cpu_idle_time(j, &cur_wall_time);

			wall_time = (unsigned int)
				(cur_wall_time - j_dbs_info->prev_cpu_wall);
			j_dbs_info->prev_cpu_wall = cur_wall_time;

			idle_time = (unsigned int)
				(cur_idle_time - j_dbs_info->prev_cpu_idle);
			j_dbs_info->prev_cpu_idle = cur_idle_time;

			if (unlikely(!wall_time || wall_time < idle_time))
				continue;

			load = 100 * (wall_time - idle_time) / wall_time;

			if (load > max_load)
				max_load = load;
		}

		/* Check for frequency increase */
		if (max_load > dbs_tuners_ins.up_threshold) {
		this_dbs_info->down_skip = 0;

			/* if we are already at full speed then break out early */
			if (this_dbs_info->requested_freq == policy->max)
				return;

			freq_target = (dbs_tuners_ins.freq_step * policy->max) / 100;

			/* max freq cannot be less than 100. But who knows.... */
			if (unlikely(freq_target == 0))
				freq_target = 5;
		
			this_dbs_info->requested_freq += freq_target;
			if (suspended) {
				if (this_dbs_info->requested_freq > dbs_tuners_ins.suspend_max_freq) {
					this_dbs_info->requested_freq = dbs_tuners_ins.suspend_max_freq;
				} else if (this_dbs_info->requested_freq > policy->max) {
					this_dbs_info->requested_freq = policy->max;
				}
			}

			__cpufreq_driver_target(policy, this_dbs_info->requested_freq,
				CPUFREQ_RELATION_H);
			return;
		}

		/*
		 * The optimal frequency is the frequency that is the lowest that
		 * can support the current CPU usage without triggering the up
		 * policy. To be safe, we focus 10 points under the threshold.
		 */
		if (max_load < (dbs_tuners_ins.down_threshold - 10)) {
			freq_target = (dbs_tuners_ins.freq_step * policy->max) / 100;

			this_dbs_info->requested_freq -= freq_target;
			if (this_dbs_info->requested_freq < policy->min)
				this_dbs_info->requested_freq = policy->min;
			
			/*
			 * if we cannot reduce the frequency anymore, break out early
			 */
			if (policy->cur == policy->min)
				return;

			__cpufreq_driver_target(policy, this_dbs_info->requested_freq,
					CPUFREQ_RELATION_H);
			return;
		}
	}
}

static void do_dbs_timer(struct work_struct *work)
{
	struct cpu_dbs_info_s *dbs_info =
		container_of(work, struct cpu_dbs_info_s, work.work);
	unsigned int cpu = dbs_info->cpu;

	/* We want all CPUs to do sampling nearly on same jiffy */
	int delay = usecs_to_jiffies(dbs_tuners_ins.sampling_rate);

	delay -= jiffies % delay;

	mutex_lock(&dbs_info->timer_mutex);

	phantom_input_boost(dbs_info);
	dbs_check_cpu(dbs_info);
	dbs_check_cpu_suspend(dbs_info);

	queue_delayed_work_on(cpu, dbs_wq, &dbs_info->work, delay);
	mutex_unlock(&dbs_info->timer_mutex);
}

static inline void dbs_timer_init(struct cpu_dbs_info_s *dbs_info)
{
	/* We want all CPUs to do sampling nearly on same jiffy */
	int delay = usecs_to_jiffies(dbs_tuners_ins.sampling_rate);
	delay -= jiffies % delay;

	dbs_info->enable = 1;
	INIT_DELAYED_WORK_DEFERRABLE(&dbs_info->work, do_dbs_timer);
	queue_delayed_work_on(dbs_info->cpu, dbs_wq, &dbs_info->work, delay);
}

static inline void dbs_timer_exit(struct cpu_dbs_info_s *dbs_info)
{
	dbs_info->enable = 0;
	cancel_delayed_work_sync(&dbs_info->work);
}

static void phantom_input_event(struct input_handle *handle,
                unsigned int type, unsigned int code, int value)
{
	if (type == EV_SYN && code == SYN_REPORT) {
		u64 now;

		/* 
		 * If input just occured, don't enable input again but wait 
		 * [input_boost_time] seconds 
		 */
		now = ktime_to_ms(ktime_get());
		if (now - last_input_time > dbs_tuners_ins.input_boost_time)
			input = true;

		last_input_time = ktime_to_ms(ktime_get());
	}

}

static int phantom_input_connect(struct input_handler *handler,
                struct input_dev *dev, const struct input_device_id *id)
{
	struct input_handle *handle;
	int error;

	handle = kzalloc(sizeof(struct input_handle), GFP_KERNEL);
	if (!handle)
		return -ENOMEM;

	handle->dev = dev;
	handle->handler = handler;
	handle->name = "cpufreq";

	error = input_register_handle(handle);
	if (error)
		goto err2;

	error = input_open_device(handle);
	if (error)
		goto err1;

	return 0;
err1:
	input_unregister_handle(handle);
err2:
	kfree(handle);
	return error;
}

static void phantom_input_disconnect(struct input_handle *handle)
{
	input_close_device(handle);
	input_unregister_handle(handle);
	kfree(handle);
}

static const struct input_device_id phantom_ids[] = {
	/* multi-touch touchscreen */
	{
		.flags = INPUT_DEVICE_ID_MATCH_EVBIT |
				INPUT_DEVICE_ID_MATCH_ABSBIT,
		.evbit = { BIT_MASK(EV_ABS) },
		.absbit = { [BIT_WORD(ABS_MT_POSITION_X)] =
				BIT_MASK(ABS_MT_POSITION_X) |
				BIT_MASK(ABS_MT_POSITION_Y) },
	},
	/* touchpad */
	{
		.flags = INPUT_DEVICE_ID_MATCH_KEYBIT |
			INPUT_DEVICE_ID_MATCH_ABSBIT,
		.keybit = { [BIT_WORD(BTN_TOUCH)] = BIT_MASK(BTN_TOUCH) },
		.absbit = { [BIT_WORD(ABS_X)] =
			BIT_MASK(ABS_X) | BIT_MASK(ABS_Y) },
	},
	/* Keypad */
	{
		.flags = INPUT_DEVICE_ID_MATCH_EVBIT,
		.evbit = { BIT_MASK(EV_KEY) },
	},
	{ },
};

static struct input_handler phantom_input_handler = {
	.event          = phantom_input_event,
	.connect        = phantom_input_connect,
	.disconnect     = phantom_input_disconnect,
	.name           = "cpufreq_phantom",
	.id_table       = phantom_ids,
};

static int cpufreq_governor_dbs(struct cpufreq_policy *policy,
				   unsigned int event)
{
	unsigned int cpu = policy->cpu;
	struct cpu_dbs_info_s *this_dbs_info;
	unsigned int j;
	int rc;

	this_dbs_info = &per_cpu(cs_cpu_dbs_info, cpu);

	switch (event) {
	case CPUFREQ_GOV_START:
		if ((!cpu_online(cpu)) || (!policy->cur))
			return -EINVAL;

		mutex_lock(&dbs_mutex);

		for_each_cpu(j, policy->cpus) {
			struct cpu_dbs_info_s *j_dbs_info;
			j_dbs_info = &per_cpu(cs_cpu_dbs_info, j);
			j_dbs_info->cur_policy = policy;

			j_dbs_info->prev_cpu_idle = get_cpu_idle_time(j,
						&j_dbs_info->prev_cpu_wall);
		}
		this_dbs_info->down_skip = 0;
		this_dbs_info->requested_freq = policy->cur;
		this_dbs_info->freq_table = cpufreq_frequency_get_table(cpu);

		mutex_init(&this_dbs_info->timer_mutex);
		dbs_enable++;
		/*
		 * Start the timerschedule work, when this governor
		 * is used for first time
		 */
		if (dbs_enable == 1) {
			unsigned int latency;
			/* policy latency is in nS. Convert it to uS first */
			latency = policy->cpuinfo.transition_latency / 1000;
			if (latency == 0)
				latency = 1;

			rc = sysfs_create_group(cpufreq_global_kobject,
						&dbs_attr_group);
			if (rc) {
				mutex_unlock(&dbs_mutex);
				return rc;
			}

			/*
			 * phantom does not implement micro like ondemand
			 * governor, thus we are bound to jiffes/HZ
			 */
			min_sampling_rate =
				MIN_SAMPLING_RATE_RATIO * jiffies_to_usecs(10);
			/* Bring kernel and HW constraints together */
			min_sampling_rate = max(min_sampling_rate,
					MIN_LATENCY_MULTIPLIER * latency);
			dbs_tuners_ins.sampling_rate =
				max(min_sampling_rate,
				    latency * LATENCY_MULTIPLIER);

			cpufreq_register_notifier(
					&dbs_cpufreq_notifier_block,
					CPUFREQ_TRANSITION_NOTIFIER);
		}
		if (!cpu)
			rc = input_register_handler(&phantom_input_handler);
		mutex_unlock(&dbs_mutex);

		dbs_timer_init(this_dbs_info);

		break;

	case CPUFREQ_GOV_STOP:
		dbs_timer_exit(this_dbs_info);

		mutex_lock(&dbs_mutex);
		dbs_enable--;
		mutex_destroy(&this_dbs_info->timer_mutex);

		if (!cpu)
			input_unregister_handler(&phantom_input_handler);
		/*
		 * Stop the timerschedule work, when this governor
		 * is used for first time
		 */
		if (dbs_enable == 0)
			cpufreq_unregister_notifier(
					&dbs_cpufreq_notifier_block,
					CPUFREQ_TRANSITION_NOTIFIER);

		mutex_unlock(&dbs_mutex);
		if (!dbs_enable)
			sysfs_remove_group(cpufreq_global_kobject,
					   &dbs_attr_group);

		break;

	case CPUFREQ_GOV_LIMITS:
		mutex_lock(&this_dbs_info->timer_mutex);
		if (policy->max < this_dbs_info->cur_policy->cur)
			__cpufreq_driver_target(
					this_dbs_info->cur_policy,
					policy->max, CPUFREQ_RELATION_H);
		else if (policy->min > this_dbs_info->cur_policy->cur)
			__cpufreq_driver_target(
					this_dbs_info->cur_policy,
					policy->min, CPUFREQ_RELATION_L);
		mutex_unlock(&this_dbs_info->timer_mutex);

		break;
	}
	return 0;
}

struct cpufreq_governor cpufreq_gov_phantom = {
	.name			= "phantom",
	.governor		= cpufreq_governor_dbs,
	.max_transition_latency	= TRANSITION_LATENCY_LIMIT,
	.owner			= THIS_MODULE,
};

static void phantom_early_suspend(struct early_suspend *handler) {
	if (suspended) // already suspended so nothing to do
		return;
	suspended = 1;
}

static void phantom_late_resume(struct early_suspend *handler) {
	if (!suspended) // already not suspended so nothing to do
		return;
	suspended = 0;
}

static struct early_suspend phantom_power_suspend = {
	.suspend = phantom_early_suspend,
	.resume = phantom_late_resume,
};

static int __init cpufreq_gov_dbs_init(void)
{
	dbs_wq = alloc_workqueue("phantom_dbs_wq", WQ_HIGHPRI, 0);
	if (!dbs_wq) {
		printk(KERN_ERR "Failed to create phantom_dbs_wq workqueue\n");
		return -EFAULT;
	}

	register_early_suspend(&phantom_power_suspend);

	return cpufreq_register_governor(&cpufreq_gov_phantom);
}

static void __exit cpufreq_gov_dbs_exit(void)
{
	cpufreq_unregister_governor(&cpufreq_gov_phantom);
	destroy_workqueue(dbs_wq);
}


MODULE_AUTHOR("Jake van der Putten <jakevdputten@gmail.com>");
MODULE_DESCRIPTION("'cpufreq_phantom' - A dynamic cpufreq governor for "
		"Low Latency Frequency Transition capable processors ");
MODULE_LICENSE("GPL");

#ifdef CONFIG_CPU_FREQ_DEFAULT_GOV_PHANTOM
fs_initcall(cpufreq_gov_dbs_init);
#else
module_init(cpufreq_gov_dbs_init);
#endif
module_exit(cpufreq_gov_dbs_exit);
