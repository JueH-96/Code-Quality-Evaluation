class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        ans = n
        for i in range(n):
            for j in range(i + 1, n):
                if (int(num[i]) * 10 + int(num[j])) % 25 == 0:
                    ans = min(ans, n - j - 1 + j - i - 1 + i)

        cnt = 0
        for i in range(n - 1, -1, -1):
            if num[i] == '0':
                cnt += 1
                break
            else:
                cnt += 1
        
        if cnt < n:
            ans = min(ans, cnt)
        else:
            ans = min(ans, n)

        return ans