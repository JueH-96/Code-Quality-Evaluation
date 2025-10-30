class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Convert possible array to 1 and -1
        arr = [1 if x else -1 for x in possible]
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + arr[i]
        total = prefix[-1]
        # Check each possible k from 1 to n-1
        for k in range(1, n):
            if 2 * prefix[k] > total:
                return k
        return -1