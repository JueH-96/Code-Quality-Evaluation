class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairs = [(nums[i], nums[n - 1 - i]) for i in range(n // 2)]
        
        def cost(a, b, X):
            if X == abs(a - b):
                return 0
            elif X <= max(k - b, b, k - a, a):
                return 1
            else:
                return 2
        
        # Collect candidate X values
        candidates = set()
        for a, b in pairs:
            candidates.add(abs(a - b))
            max_one_change = max(k - b, b, k - a, a)
            candidates.add(max_one_change + 1)
        
        # Only consider valid X values
        candidates = {X for X in candidates if 0 <= X <= k}
        
        min_changes = float('inf')
        for X in candidates:
            total_changes = sum(cost(a, b, X) for a, b in pairs)
            min_changes = min(min_changes, total_changes)
        
        return min_changes