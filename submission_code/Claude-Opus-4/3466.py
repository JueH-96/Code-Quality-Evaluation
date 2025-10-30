class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            current_and = nums[i]
            
            for j in range(i, n):
                if i != j:
                    current_and &= nums[j]
                
                # If current_and has 0 where k has 1, we can't get k anymore
                if (current_and & k) != current_and:
                    break
                
                if current_and == k:
                    count += 1
        
        return count