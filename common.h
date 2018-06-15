/*
 * common.h
 *
 *  Created on: Dec 11, 2013
 *      Author: ryan
 */

#ifndef COMMON_H_
#define COMMON_H_

#define EST
#define CHECKS	0.3

#include <sys/time.h>
#include <stdint.h>

#define TIMER_T                         struct timeval
#define TIMER_READ(time)                gettimeofday(&(time), NULL)
#define TIMER_DIFF_SECONDS(start, stop) \
    (((double)(stop.tv_sec)  + (double)(stop.tv_usec / 1000000.0)) - \
     ((double)(start.tv_sec) + (double)(start.tv_usec / 1000000.0)))

inline uint64_t tick() {
	uint32_t tmp[2];
	__asm__ ("rdtsc" : "=a" (tmp[1]), "=d" (tmp[0]) : "c" (0x10) );
	return (((uint64_t) tmp[0]) << 32) | tmp[1];
}

inline float vdistance_(const float* v1, const float* v2, size_t len) {
	float dist = 0;
//#pragma simd reduction(+:dist)
	for (size_t i = 0; i < len; ++i) {
		float tmp = v1[i] - v2[i];
		dist += tmp * tmp;
	}
	return dist;
}

#endif /* COMMON_H_ */
