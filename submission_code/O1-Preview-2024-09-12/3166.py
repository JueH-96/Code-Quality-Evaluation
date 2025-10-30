class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        def is_possible(k, counts, n):
            s_min = n // k
            extra = n % k
            s_max = s_min + 1 if extra > 0 else s_min
            total_groups_needed = 0
            for count in counts:
                groups_needed = (count + s_max - 1) // s_max
                total_groups_needed += groups_needed
                if total_groups_needed > k:
                    return False
            return True

        n = len(nums)
        counts = list(Counter(nums).values())
        left, right = 1, n
        answer = n
        while left <= right:
            mid = (left + right) // 2
            if is_possible(mid, counts, n):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer