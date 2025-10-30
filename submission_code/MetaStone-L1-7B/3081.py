import bisect

def minLengthAfterRemovals(nums: list[int]) -> int:
    n = len(nums)
    count = 0
    i = 0
    while i < n:
        target = nums[i]
        j = bisect.bisect_right(nums, target, i + 1, n)
        if j < n:
            count += 1
            i = j + 1
        else:
            i += 1
    return n - 2 * count