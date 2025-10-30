import sys
from collections import defaultdict

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)

        # If k is large enough (>= 26), any segment has at most 26 distinct
        # characters. The entire string will always form a single partition.
        # Changing one character cannot increase the number of partitions beyond 1.
        if k >= 26:
             return 1

        # dp[mask][changed] = max partitions ending at current index i-1
        # where the last partition ends at index i-1, has distinct chars `mask`,
        # and `changed` is 0 if no change used, 1 if one change used globally up to index i-1.

        # dp states for index i-1
        # We use a dictionary where keys are masks (int) and values are another dictionary
        # mapping changed status (0 or 1) to max partitions.
        # Initial state: Before index 0 (at index -1), there are 0 partitions.
        # An implicit empty partition ends at index -1. Mask 0, changed 0.
        dp = {0: {0: 0}} 

        # Iterate from index i = 0 to n-1
        for i in range(n):
            # next_dp stores states for index i
            next_dp = defaultdict(lambda: defaultdict(lambda: -sys.maxsize))

            # Iterate over states ending at i-1 (from dp)
            # prev_mask: distinct chars in partition ending at i-1
            # prev_changed: global change status up to i-1
            # p: max partitions ending at i-1
            for prev_mask, changed_states in dp.items():
                for prev_changed, p in changed_states.items():
                    if p == -sys.maxsize: continue

                    # Option A: s[i] extends the partition ending at i-1.
                    # The partition covering i ends at i.
                    # This is valid only if adding s[i] does not exceed k distinct characters.
                    char_mask_s_i = 1 << (ord(s[i]) - ord('a'))
                    curr_mask_s_i = prev_mask | char_mask_s_i
                    
                    # Count distinct characters in the resulting mask curr_mask_s_i
                    distinct_count_s_i = 0
                    temp_mask = curr_mask_s_i
                    while temp_mask > 0:
                        if temp_mask & 1:
                            distinct_count_s_i += 1
                        temp_mask >>= 1

                    if distinct_count_s_i <= k:
                        # Valid continuation. The partition continues and ends at index i.
                        # The number of partitions remains p.
                        # Update the state for a partition ending at index i with mask curr_mask_s_i
                        next_dp[curr_mask_s_i][prev_changed] = max(next_dp[curr_mask_s_i][prev_changed], p)
                    else:
                        # Adding s[i] exceeds k. The partition must end at i-1.
                        # A new partition starts at i, and this new partition ends at i.
                        # This new partition consists solely of s[i].
                        # The number of partitions becomes p + 1 (the partition ending at i-1 + the new one at i).
                        start_mask_s_i = char_mask_s_i
                        next_dp[start_mask_s_i][prev_changed] = max(next_dp[start_mask_s_i][prev_changed], p + 1)

                    # Option B: s[i] is changed to c, extends the partition ending at i-1.
                    # The partition covering i ends at i.
                    # This requires the global change not to have been used before index i (prev_changed == 0).
                    if prev_changed == 0:
                        for c_ord in range(26):
                            char_mask_c = 1 << c_ord
                            curr_mask_c = prev_mask | char_mask_c
                            next_changed = 1 # The change is now used globally

                            # Count distinct characters in the resulting mask curr_mask_c
                            distinct_count_c = 0
                            temp_mask = curr_mask_c
                            while temp_mask > 0:
                                if temp_mask & 1:
                                    distinct_count_c += 1
                                temp_mask >>= 1

                            if distinct_count_c <= k:
                                # Valid continuation. The partition continues and ends at index i.
                                # The number of partitions remains p.
                                # Update the state for a partition ending at index i with mask curr_mask_c and changed=1
                                next_dp[curr_mask_c][next_changed] = max(next_dp[curr_mask_c][next_changed], p)
                            else:
                                # Adding c exceeds k. The partition must end at i-1.
                                # A new partition starts at i, and this new partition ends at i.
                                # This new partition consists solely of c.
                                # The number of partitions becomes p + 1.
                                start_mask_c = char_mask_c
                                next_dp[start_mask_c][next_changed] = max(next_dp[start_mask_c][next_changed], p + 1)
            
            # After processing all characters up to index i, next_dp holds the states ending at index i.
            dp = next_dp 

        # The max value in the final dp (states ending at index n-1) 
        # represents the maximum number of partitions for the entire string s.
        max_partitions = 0
        for mask, changed_states in dp.items():
            for changed, p in changed_states.items():
                if p != -sys.maxsize:
                    max_partitions = max(max_partitions, p)

        return max_partitions