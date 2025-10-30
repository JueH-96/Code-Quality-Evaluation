class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ones = [0] * (n + 1)
        zeros = [0] * (n + 1)
        for i in range(1, n + 1):
            if s[i - 1] == '0':
                zeros[i] = zeros[i - 1] + 1
                ones[i] = ones[i - 1]
            else:
                ones[i] = ones[i - 1] + 1
                zeros[i] = zeros[i - 1]
        count = 0
        for i in range(1, n + 1):
            l, r = i, n
            while l <= r:
                mid = (l + r) // 2
                if zeros[mid] - zeros[i - 1] < (ones[mid] - ones[i - 1]) ** 0.5:
                    l = mid + 1
                else:
                    r = mid - 1
            count += n - l + 1
        return count