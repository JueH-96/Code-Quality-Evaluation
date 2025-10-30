class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        min_substr = ''

        for i in range(n):
            if i + k > n:
                break

            freq = [0]*10
            flag = True
            for j in range(i, i+k):
                if s[j] == '0':
                    if freq[1] > 0 or freq[2] > 0 or freq[3] > 0 or freq[4] > 0 or freq[5] > 0 or freq[6] > 0 or freq[7] > 0 or freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[1] += 1
                elif s[j] == '1':
                    if freq[2] > 0 or freq[3] > 0 or freq[4] > 0 or freq[5] > 0 or freq[6] > 0 or freq[7] > 0 or freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[2] += 1
                elif s[j] == '2':
                    if freq[3] > 0 or freq[4] > 0 or freq[5] > 0 or freq[6] > 0 or freq[7] > 0 or freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[3] += 1
                elif s[j] == '3':
                    if freq[4] > 0 or freq[5] > 0 or freq[6] > 0 or freq[7] > 0 or freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[4] += 1
                elif s[j] == '4':
                    if freq[5] > 0 or freq[6] > 0 or freq[7] > 0 or freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[5] += 1
                elif s[j] == '5':
                    if freq[6] > 0 or freq[7] > 0 or freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[6] += 1
                elif s[j] == '6':
                    if freq[7] > 0 or freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[7] += 1
                elif s[j] == '7':
                    if freq[8] > 0 or freq[9] > 0:
                        flag = False
                        break
                    freq[8] += 1
                elif s[j] == '8':
                    if freq[9] > 0:
                        flag = False
                        break
                    freq[9] += 1
                else:
                    flag = False
                    break

            if flag and sum(freq) == k:
                substr = s[i:i+k]
                if min_len > len(substr):
                    min_len = len(substr)
                    min_substr = substr
                elif min_len == len(substr):
                    min_substr = min(min_substr, substr)

        return min_substr