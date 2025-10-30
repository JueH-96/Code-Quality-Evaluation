def min_absolute_difference(nums, x):
    min_diff = float('inf')
    n = len(nums)
    if x == 0:
        return 0
    if x >= n:
        return 0
    for i in range(n - x + 1):
        window = nums[i:i+x]
        current_min = min(window)
        current_max = max(window)
        diff1 = abs(nums[i] - current_min)
        diff2 = abs(nums[i] - current_max)
        min_diff = min(min_diff, diff1, diff2)
    return min_diff