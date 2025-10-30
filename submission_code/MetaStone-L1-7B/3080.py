def maxSubarrays(nums: List[int]) -> int:
    zero_runs = 0
    non_zero_between = 0
    prev_zero = False

    for num in nums:
        if num == 0:
            if not prev_zero:
                zero_runs += 1
                prev_zero = True
            else:
                prev_zero = True
        else:
            if prev_zero:
                non_zero_between += 1
                prev_zero = False

    # Check if the array ends with a non-zero run that is after a zero run
    if not prev_zero:
        non_zero_between += 1

    return zero_runs + non_zero_between