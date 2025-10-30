class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        delta = [0] * (n + 1)  # We use an extra space to handle range updates easily
        
        # Apply the difference array technique for range increment
        for l, r in queries:
            delta[l] += 1
            if r + 1 < n:
                delta[r + 1] -= 1
        
        # Calculate the actual number of decrements applied to each position
        current_decrement = 0
        for i in range(n):
            current_decrement += delta[i]
            nums[i] -= current_decrement
        
        # Check if all elements are zero
        return all(x == 0 for x in nums)