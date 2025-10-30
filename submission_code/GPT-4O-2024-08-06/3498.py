class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        n = len(nums)
        half_n = n // 2
        changes_needed = 0

        # This dictionary will store the frequency of each pair (min, max) in the array
        pair_count = defaultdict(int)

        # Iterate over the first half of the array
        for i in range(half_n):
            a, b = nums[i], nums[n - i - 1]
            min_val, max_val = min(a, b), max(a, b)
            pair_count[(min_val, max_val)] += 1

        # We need to find the most common pair (min, max) that can be transformed into
        # a valid pair (x, x + X) or (x + X, x) with the least changes
        max_common_pairs = 0

        # Check all possible values of X
        for X in range(k + 1):
            current_common_pairs = 0
            for (min_val, max_val), count in pair_count.items():
                if max_val - min_val == X:
                    current_common_pairs += count
            max_common_pairs = max(max_common_pairs, current_common_pairs)

        # The minimum changes needed is the total pairs minus the maximum common pairs
        changes_needed = half_n - max_common_pairs

        return changes_needed