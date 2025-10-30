from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix ORs
        # prefix_or[i] stores nums[0] | nums[1] | ... | nums[i]
        prefix_or = [0] * n
        # The constraint 1 <= n <= 10^5 guarantees n > 0
        prefix_or[0] = nums[0]
        for i in range(1, n):
            prefix_or[i] = prefix_or[i-1] | nums[i]
            
        # Calculate suffix ORs
        # suffix_or[i] stores nums[i] | nums[i+1] | ... | nums[n-1]
        suffix_or = [0] * n
        # The constraint 1 <= n <= 10^5 guarantees n > 0
        suffix_or[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_or[i] = nums[i] | suffix_or[i+1]

        # Calculate 2^k
        two_pow_k = 1 << k
        
        max_or = 0

        # Iterate through each element nums[i] and consider applying all k operations to it.
        # The resulting OR value in this case would be:
        # (nums[i] * 2^k) | (OR of elements before index i) | (OR of elements after index i)
        for i in range(n):
            current_num_shifted = nums[i] * two_pow_k
            
            # Calculate the OR of elements excluding nums[i].
            # This is the OR of elements nums[0...i-1] and nums[i+1...n-1].
            others_or = 0
            
            # OR of elements nums[0]...nums[i-1]
            if i > 0:
                others_or |= prefix_or[i-1]
                
            # OR of elements nums[i+1]...nums[n-1]
            if i < n - 1:
                others_or |= suffix_or[i+1]
                
            # The total OR value if nums[i] is the one receiving k operations
            current_total_or = current_num_shifted | others_or
            
            # Update the maximum OR found so far
            max_or = max(max_or, current_total_or)
            
        return max_or