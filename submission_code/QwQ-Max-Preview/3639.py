class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        sum_queries = 0
        # Initialize the difference array to compute coverage
        delta = [0] * (n + 2)
        
        for l, r in queries:
            delta[l] += 1
            if r + 1 < n:
                delta[r + 1] -= 1
            sum_queries += r - l + 1
        
        # Compute the prefix sum to get the coverage count for each index
        current = 0
        coverage = []
        for j in range(n):
            current += delta[j]
            coverage.append(current)
        
        # Check if each element's coverage is at least as much as its value
        for j in range(n):
            if coverage[j] < nums[j]:
                return False
        
        # Check if the total required decrements is feasible
        total_required = sum(nums)
        return total_required <= sum_queries