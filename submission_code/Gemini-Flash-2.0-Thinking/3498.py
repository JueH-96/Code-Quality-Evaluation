class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_changes = n

        for X in range(k + 1):
            current_changes = 0
            for i in range(n // 2):
                a = nums[i]
                b = nums[n - 1 - i]

                if abs(a - b) == X:
                    continue
                else:
                    can_change_a = (0 <= b + X <= k) or (0 <= b - X <= k)
                    can_change_b = (0 <= a + X <= k) or (0 <= a - X <= k)

                    if can_change_a or can_change_b:
                        current_changes += 1
                    else:
                        current_changes += 2

            min_changes = min(min_changes, current_changes)

        return min_changes