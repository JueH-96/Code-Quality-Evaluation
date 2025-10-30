class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        
        # Use difference array to count how many queries affect each position
        diff = [0] * (n + 1)
        
        # For each query [l, r], we can decrement any position in range [l, r]
        # This means each position in [l, r] gets +1 potential decrements
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1
        
        # Convert difference array to actual counts
        count = 0
        for i in range(n):
            count += diff[i]
            # Check if we can decrement nums[i] enough times to make it 0
            if count < nums[i]:
                return False
        
        return True