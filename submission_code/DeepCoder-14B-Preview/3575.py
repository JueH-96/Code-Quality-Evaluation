class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        def max_or_second_part(nums, start, k):
            dp = {0: {0}}
            for j in range(start, len(nums)):
                current_num = nums[j]
                new_dp = {}
                for t in dp:
                    for o in dp[t]:
                        if t < k:
                            new_t = t + 1
                            new_o = o | current_num
                            if new_t not in new_dp:
                                new_dp[new_t] = set()
                            new_dp[new_t].add(new_o)
                # Merge new_dp into the main dp
                for t in new_dp:
                    if t not in dp:
                        dp[t] = set()
                    for o_val in new_dp[t]:
                        dp[t].add(o_val)
            if k in dp:
                return max(dp[k])
            else:
                return 0
        
        # Process the first part
        first_part_dp = {(0, -1): {0}}
        for j in range(len(nums)):
            current_num = nums[j]
            new_dp = {}
            for (current_m, current_last_i) in list(first_part_dp.keys()):
                current_or_set = first_part_dp[(current_m, current_last_i)]
                for o in current_or_set:
                    if current_m < k:
                        new_m = current_m + 1
                        new_last_i = j
                        new_o = o | current_num
                        key = (new_m, new_last_i)
                        if key not in new_dp:
                            new_dp[key] = set()
                        new_dp[key].add(new_o)
            # Merge new_dp into the main first_part_dp
            for key in new_dp:
                if key not in first_part_dp:
                    first_part_dp[key] = set()
                for o_val in new_dp[key]:
                    first_part_dp[key].add(o_val)
        
        # Collect all possible states where m == k
        possible_first_states = []
        for (m, last_i) in first_part_dp:
            if m == k:
                for o in first_part_dp[(m, last_i)]:
                    possible_first_states.append((last_i, o))
        
        max_xor = 0
        for (last_i, o1) in possible_first_states:
            start = last_i + 1
            if start >= len(nums):
                continue  # No elements left for the second part
            o2 = max_or_second_part(nums, start, k)
            if o2 == 0:
                continue  # Not possible to select k elements
            current_xor = o1 ^ o2
            if current_xor > max_xor:
                max_xor = current_xor
        
        return max_xor