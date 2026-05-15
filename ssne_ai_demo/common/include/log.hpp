/*
 * @Filename: log.hpp
 * @Description: SSNE AI Demo 统一调试日志宏定义
 */
#pragma once

#include <cstdio>

#define ENABLE_DEBUG_LOG 0

#if ENABLE_DEBUG_LOG
    #define LOG_DEBUG(fmt_str, ...) \
        printf("[DEBUG] " fmt_str, ##__VA_ARGS__)
#else
    #define LOG_DEBUG(fmt_str, ...)
#endif