class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        n = len(nums)
        total_pairs = n // 2
        freq = defaultdict(int)
        pairs = []
        for i in range(n // 2):
            a = nums[i]
            b = nums[n - i - 1]
            d = abs(a - b)
            freq[d] +=1
            pairs.append((a, b, d))
        min_changes = float('inf')
        for D_max in freq.keys():
            changes = 0
            for a, b, d in pairs:
                if d == D_max:
                    continue
                # Try changing a to b +/- D_max
                change = 2  # default cost
                a_options = [b + D_max, b - D_max]
                for a_new in a_options:
                    if 0 <= a_new <= k:
                        change = 1
                        break
                b_options = [a + D_max, a - D_max]
                for b_new in b_options:
                    if 0 <= b_new <= k:
                        change = 1
                        break
                changes += change
            min_changes = min(min_changes, changes)
        return min_changes