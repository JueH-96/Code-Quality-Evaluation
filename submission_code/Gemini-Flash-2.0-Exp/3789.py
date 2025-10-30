class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def count_subarrays(pairs):
            count = 0
            for i in range(1, n + 1):
                for j in range(i, n + 1):
                    subarray = list(range(i, j + 1))
                    valid = True
                    for a, b in pairs:
                        if a in subarray and b in subarray:
                            valid = False
                            break
                    if valid:
                        count += 1
            return count

        max_count = 0
        for i in range(len(conflictingPairs)):
            temp_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            max_count = max(max_count, count_subarrays(temp_pairs))

        return max_count