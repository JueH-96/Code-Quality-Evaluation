class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        result = 0
        
        # For each position that could be the middle of a subsequence
        for mid in range(2, n - 2):
            mid_val = nums[mid]
            
            # Count occurrences of mid_val before and after position mid
            left_count = {}
            right_count = {}
            
            # Count elements to the left
            for i in range(mid):
                if nums[i] not in left_count:
                    left_count[nums[i]] = 0
                left_count[nums[i]] += 1
            
            # Count elements to the right
            for i in range(mid + 1, n):
                if nums[i] not in right_count:
                    right_count[nums[i]] = 0
                right_count[nums[i]] += 1
            
            # For each way to pick 2 elements from left and 2 from right
            # We need to ensure mid_val appears most frequently
            
            # Get all unique values that could be in our subsequence
            all_vals = set()
            all_vals.add(mid_val)
            all_vals.update(left_count.keys())
            all_vals.update(right_count.keys())
            
            # For each combination of frequencies of mid_val on left and right
            for left_mid_count in range(min(3, left_count.get(mid_val, 0) + 1)):
                for right_mid_count in range(min(3, right_count.get(mid_val, 0) + 1)):
                    total_mid_count = left_mid_count + right_mid_count + 1  # +1 for the middle position
                    
                    # Count valid ways to pick remaining elements
                    valid_ways = 0
                    
                    # We need to pick (2 - left_mid_count) more from left
                    # and (2 - right_mid_count) more from right
                    left_needed = 2 - left_mid_count
                    right_needed = 2 - right_mid_count
                    
                    # Dynamic programming to count valid combinations
                    # dp[val][count] = ways to achieve this count for val
                    from collections import defaultdict
                    
                    # Count ways to pick from left
                    left_ways = defaultdict(int)
                    left_ways[()] = 1  # empty selection
                    
                    for val in left_count:
                        if val == mid_val:
                            continue
                        new_ways = defaultdict(int)
                        for state, ways in left_ways.items():
                            new_ways[state] = ways
                            # Try adding this value
                            for cnt in range(1, min(left_needed + 1, left_count[val] + 1)):
                                if len(state) + cnt <= left_needed:
                                    new_state = tuple(sorted(list(state) + [val] * cnt))
                                    new_ways[new_state] = (new_ways[new_state] + ways) % MOD
                        left_ways = new_ways
                    
                    # Similar for right
                    right_ways = defaultdict(int)
                    right_ways[()] = 1
                    
                    for val in right_count:
                        if val == mid_val:
                            continue
                        new_ways = defaultdict(int)
                        for state, ways in right_ways.items():
                            new_ways[state] = ways
                            for cnt in range(1, min(right_needed + 1, right_count[val] + 1)):
                                if len(state) + cnt <= right_needed:
                                    new_state = tuple(sorted(list(state) + [val] * cnt))
                                    new_ways[new_state] = (new_ways[new_state] + ways) % MOD
                        right_ways = new_ways
                    
                    # Combine left and right selections
                    for left_state, l_ways in left_ways.items():
                        if len(left_state) == left_needed:
                            for right_state, r_ways in right_ways.items():
                                if len(right_state) == right_needed:
                                    # Check if mid_val is the unique mode
                                    freq_count = defaultdict(int)
                                    freq_count[mid_val] = total_mid_count
                                    
                                    for val in left_state:
                                        freq_count[val] += 1
                                    for val in right_state:
                                        freq_count[val] += 1
                                    
                                    # Check if mid_val has the highest frequency
                                    max_freq = max(freq_count.values())
                                    mode_count = sum(1 for f in freq_count.values() if f == max_freq)
                                    
                                    if freq_count[mid_val] == max_freq and mode_count == 1:
                                        # Count ways to pick specific positions
                                        # Choose left_mid_count positions from left for mid_val
                                        from math import comb
                                        left_mid_ways = comb(left_count.get(mid_val, 0), left_mid_count)
                                        right_mid_ways = comb(right_count.get(mid_val, 0), right_mid_count)
                                        
                                        # Multiply all ways
                                        total_ways = (l_ways * r_ways * left_mid_ways * right_mid_ways) % MOD
                                        result = (result + total_ways) % MOD
        
        return result