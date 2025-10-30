class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        mod = 10**9 + 7

        from itertools import combinations

        for indices in combinations(range(n), 5):
            subsequence_indices = sorted(list(indices))
            subsequence = [nums[i] for i in subsequence_indices]

            middle_element = subsequence[2]

            counts = {}
            for x in subsequence:
                counts[x] = counts.get(x, 0) + 1

            is_unique_middle_mode = True
            middle_count = counts.get(middle_element, 0)

            if middle_count == 0:
                continue

            for element, c in counts.items():
                if element != middle_element and c >= middle_count:
                    is_unique_middle_mode = False
                    break

            if is_unique_middle_mode and middle_count > 0:
                count = (count + 1) % mod

        return count