class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # Initialize the result with 0
        k_or = 0
        
        # Iterate through 31 bits since nums[i] < 2^31
        for i in range(31):
            # Count how many numbers have the i-th bit set
            count = sum((num >> i) & 1 for num in nums)
            
            # If at least k numbers have the i-th bit set, set the i-th bit in the result
            if count >= k:
                k_or |= (1 << i)
        
        return k_or