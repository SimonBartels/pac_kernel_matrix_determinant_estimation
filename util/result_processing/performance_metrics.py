# a: default, b: stopped
pm_speed_up = lambda a, b: a / b  # speed-up
pm_fraction = lambda a, b: b / a  # fraction of original time
pm_time_saved = lambda a, b: (a - b) / a  # fraction of time saved
pm_abs_time_saved = lambda a, b: a - b
pm_abs_time_add = lambda a, b: b - a  # additional time
