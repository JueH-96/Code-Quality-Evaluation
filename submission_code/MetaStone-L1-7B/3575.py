def maximal_value(nums, k):
    from collections import defaultdict

    dp = set()
    dp.add((0, 0, 0, 0))

    for num in nums:
        new_dp = set()
        for state in dp:
            s_size, t_size, s_or, t_or = state
            # Option 1: add to S
            if s_size < k:
                new_s_size = s_size + 1
                new_s_or = s_or | num
                new_state = (new_s_size, t_size, new_s_or, t_or)
                new_dp.add(new_state)
            # Option 2: add to T
            if t_size < k:
                new_t_size = t_size + 1
                new_t_or = t_or | num
                new_state = (s_size, new_t_size, s_or, new_t_or)
                new_dp.add(new_state)
            # Option 3: add to neither
            new_state = (s_size, t_size, s_or, t_or)
            new_dp.add(new_state)
        dp.update(new_dp)

    max_xor = 0
    for state in dp:
        s_size, t_size, s_or, t_or = state
        if s_size == k and t_size == k:
            current_xor = s_or ^ t_or
            if current_xor > max_xor:
                max_xor = current_xor
    return max_xor

# Example usage
nums = [2, 6, 7]
k = 2
print(maximal_value(nums, k))  # Output: 5