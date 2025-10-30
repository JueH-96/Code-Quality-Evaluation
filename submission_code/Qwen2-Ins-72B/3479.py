class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ones = [0] * (n + 1)
        for i in range(n):
            ones[i + 1] = ones[i] + int(s[i])
        
        ans = 0
        for i in range(n):
            for j in range(i, n):
                zeros = j - i + 1 - (ones[j + 1] - ones[i])
                ones_count = ones[j + 1] - ones[i]
                if ones_count >= zeros ** 2:
                    ans += 1
                else:
                    break
        return ans