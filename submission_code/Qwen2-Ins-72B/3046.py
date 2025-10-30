class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        ans = n
        cnt = Counter(num)
        if '0' in cnt and '0' in cnt and cnt['0'] >= 2:
            ans = n - 2
        if '2' in cnt and '5' in cnt:
            ans = min(ans, n - 2)
        if '5' in cnt and '0' in cnt:
            ans = min(ans, n - 2)
        if '7' in cnt and '5' in cnt:
            ans = min(ans, n - 2)
        if '5' in cnt:
            ans = min(ans, n - 1)
        if '0' in cnt:
            ans = min(ans, n - 1)
        return ans