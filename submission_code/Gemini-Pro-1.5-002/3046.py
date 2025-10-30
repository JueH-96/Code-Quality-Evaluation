class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        ans = float('inf')
        targets = ["00", "25", "50", "75"]
        for target in targets:
            i = n - 1
            j = 1
            dels = 0
            while i >= 0 and j >= 0:
                if num[i] == target[j]:
                    j -= 1
                else:
                    dels += 1
                i -= 1
            if j < 0:
                ans = min(ans, dels + i + 1)
        return ans