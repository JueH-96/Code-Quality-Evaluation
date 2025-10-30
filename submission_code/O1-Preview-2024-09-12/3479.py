class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        answer = 0
        max_zeros = 200  # Since zeros^2 > n when zeros >= 201
        for zeros_limit in range(max_zeros + 1):
            i = 0
            while i < n:
                zeros = 0
                ones = 0
                j = i
                while j < n:
                    if s[j] == '0':
                        zeros += 1
                    else:
                        ones += 1
                    if zeros > zeros_limit:
                        break
                    if zeros == zeros_limit and ones >= zeros_limit * zeros_limit:
                        answer += 1
                    j += 1
                i += 1
        return answer