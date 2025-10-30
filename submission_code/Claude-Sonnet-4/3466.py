class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            current_and = nums[i]
            
            # Check if single element subarray equals k
            if current_and == k:
                count += 1
            
            # Extend subarray from position i
            for j in range(i + 1, n):
                current_and &= nums[j]
                
                if current_and == k:
                    count += 1
                
                # If current_and & k != current_and, then we can never get k
                # because AND can only turn 1 bits to 0, never 0 bits to 1
                if (current_and & k) != current_and:
                    break
        
        return count