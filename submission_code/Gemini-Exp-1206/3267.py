class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        count = {}
        for i in range(n):
            j = i
            while j < n and s[j] == s[i]:
                j += 1
            length = j - i
            for k in range(1, length + 1):
                if (s[i], k) not in count:
                    count[(s[i], k)] = 0
                count[(s[i], k)] += 1
            i = j - 1
        ans = -1
        for (char, length), freq in count.items():
            if freq >= 3:
                ans = max(ans, length)
        return ans