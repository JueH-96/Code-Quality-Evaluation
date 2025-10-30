class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_changes = n
        for X in range(k + 1):
            current_changes = 0
            for i in range(n // 2):
                a = nums[i]
                b = nums[n - i - 1]
                if abs(a - b) == X:
                    continue
                possible_1_change = False
                if 0 <= b + X <= k:
                    possible_1_change = True
                if 0 <= b - X <= k:
                    possible_1_change = True
                if 0 <= a + X <= k:
                    possible_1_change = True
                if 0 <= a - X <= k:
                    possible_1_change = True
                if possible_1_change:
                    current_changes += 1
                else:
                    current_changes += 2
            min_changes = min(min_changes, current_changes)
        return min_changes