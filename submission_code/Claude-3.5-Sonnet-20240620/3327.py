class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        n = len(ones)
        
        if k <= maxChanges:
            return min(2 * k, k + maxChanges)
        
        prefix = [0] + list(accumulate(ones))
        
        def get_cost(i, j):
            count = j - i + 1
            mid = (i + j) // 2
            return prefix[j+1] - prefix[mid+1] - (prefix[mid] - prefix[i])
        
        min_cost = float('inf')
        for i in range(n - (k - maxChanges) + 1):
            j = i + k - maxChanges - 1
            cost = get_cost(i, j)
            min_cost = min(min_cost, cost)
        
        return min_cost + 2 * maxChanges