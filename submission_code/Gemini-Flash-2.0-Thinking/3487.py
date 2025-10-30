class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        k = len(targetIndices)

        def is_subsequence(s, p):
            i = 0  # pointer for pattern
            j = 0  # pointer for source
            while i < len(p) and j < len(s):
                if s[j] == p[i]:
                    i += 1
                j += 1
            return i == len(p)

        def check(num_removals):
            if num_removals == 0:
                return is_subsequence(source, pattern)

            indices_to_remove = sorted(targetIndices[:num_removals])
            removed_indices_set = set(indices_to_remove)
            modified_source = "".join([char for i, char in enumerate(source) if i not in removed_indices_set])
            return is_subsequence(modified_source, pattern)

        low = 0
        high = k
        ans = 0

        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans