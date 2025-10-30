import collections
from typing import List

class Solution:
    """
    Calculates the number of special subsequences in the given array nums.

    A special subsequence is defined by indices (p, q, r, s) such that:
    1. p < q < r < s
    2. q - p > 1  (at least one element between p and q)
    3. r - q > 1  (at least one element between q and r)
    4. s - r > 1  (at least one element between r and s)
    5. nums[p] * nums[r] == nums[q] * nums[s]

    Constraints:
    7 <= nums.length <= 1000
    1 <= nums[i] <= 1000
    """
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        # Maximum possible value in nums is 1000. We use M = 1001 for array indexing up to 1000.
        M = 1001 

        # Precompute right_map[r]: A frequency map (array) where right_map[r][val] 
        # stores the count of indices s such that s >= r + 2 and nums[s] == val.
        # right_map[r] essentially contains frequency counts for the suffix nums[r+2:].
        right_map = [[0] * M for _ in range(n)]
        
        # temp_counts[val] will store the running count of val in nums[s] for s >= current_r + 2.
        # We iterate r downwards to build the suffix counts cumulatively.
        temp_counts = [0] * M
        # The largest possible value for r in a valid subsequence (p, q, r, s) is n-3
        # (e.g., if s=n-1, then r can be at most n-3 because s-r > 1).
        # Iterate r from n-3 down to 0.
        for r in range(n - 3, -1, -1):
            # As r decreases, the suffix range [r+2, n-1] grows by including index r+2.
            s_idx = r + 2
            # Check if the index s_idx is within the array bounds.
            if s_idx < n: 
                val_s = nums[s_idx]
                # Check if the value is within the expected range [1, 1000].
                if 1 <= val_s < M:
                    temp_counts[val_s] += 1
            
            # Store the accumulated counts for the suffix starting at r+2.
            # We create a copy of temp_counts for right_map[r].
            # This copy operation takes O(M) time per iteration. Total precomputation time: O(N*M).
            right_map[r] = temp_counts[:] 

        # Iterate through the middle indices q and r.
        # count_left[val_p] stores the count of indices p such that p <= q - 2 and nums[p] == val_p.
        # We use collections.Counter for efficient storage and iteration over the counts of elements to the left of q.
        count_left = collections.Counter() 

        # Iterate through index q. The valid range for q is [2, n - 5].
        # Minimum q is 2 (e.g., p=0, q=2).
        # Maximum q is n-5 (e.g., if r=n-3, then q can be at most n-5 because r-q > 1).
        # The loop iterates q from 2 up to n-3 to ensure q=n-5 is processed.
        for q in range(2, n - 2): 
            # Update count_left by adding nums[p] where p = q - 2.
            # This index p is always valid (>= 0) because q starts at 2.
            p_idx = q - 2
            val_p = nums[p_idx]
            # Check if the value is within the expected range [1, 1000].
            if 1 <= val_p < M:
                count_left[val_p] += 1
            
            # Optimization: If the current q is too large, no valid r can exist.
            # We need r >= q + 2. The maximum possible r is n-3.
            # So, we need q + 2 <= n - 3 => q <= n - 5.
            if q > n - 5: continue 
                
            val_q = nums[q]
            # Check if val_q is valid. Based on constraints, nums[i] >= 1, so val_q != 0.
            if not (1 <= val_q < M): continue # Safety check for value bounds.

            # Iterate through index r. The valid range for r is [q + 2, n - 3].
            # Minimum r is q+2 (because r-q > 1).
            # Maximum r is n-3 (because s >= r+2 needs s <= n-1).
            # The loop iterates r from q+2 up to n-3 (inclusive).
            for r in range(q + 2, n - 2): # The loop range goes up to index n-3.
                 
                 val_r = nums[r]
                 # Check if val_r is valid.
                 if not (1 <= val_r < M): continue # Safety check.
                 
                 # Get the precomputed counts for s >= r+2 using right_map.
                 # counts_s is the frequency array for the suffix nums[r+2:].
                 # Accessing right_map[r] takes O(1) time.
                 counts_s = right_map[r] 

                 current_qr_count = 0
                 # Iterate through the distinct values val_p found so far (p <= q-2), stored in count_left.
                 # The number of items in count_left is at most M, often much smaller.
                 for val_p, num_p in count_left.items():
                      # Calculate the product nums[p] * nums[r].
                      product = val_p * val_r
                      
                      # Check if the product is divisible by nums[q].
                      # val_q > 0 based on constraints.
                      if product % val_q == 0:
                          # Calculate the required value for nums[s].
                          target_s_value = product // val_q
                          
                          # Check if the target value for nums[s] is within the valid range [1, 1000].
                          if 1 <= target_s_value < M:
                              # Look up the count of this target_s_value in the precomputed suffix counts.
                              # This is an O(1) array lookup.
                              num_s = counts_s[target_s_value] 
                              
                              # If num_s > 0, it means there are 'num_s' valid indices s >= r + 2
                              # such that nums[s] equals the target value.
                              # Each of the 'num_p' choices for index p can be combined with each of the 'num_s' choices for index s.
                              if num_s > 0:
                                 current_qr_count += num_p * num_s
                                  
                 # Add the total count found for this specific (q, r) pair to the overall answer.
                 ans += current_qr_count

        return ans