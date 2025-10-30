class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        ans = k - 1
        for i in range(1, k):
            count = i - 1
            num = k // i
            rem = k % i
            if rem == 0:
                count += num - 1
            else:
                count += num
            ans = min(ans, count)
        return ans