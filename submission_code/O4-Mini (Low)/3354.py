class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # freq[i] = count of letter chr(i + ord('a')) seen so far
        freq = [0] * 26
        res = []
        
        for ch in s:
            if ch != '?':
                # fixed character: update freq and append
                idx = ord(ch) - ord('a')
                freq[idx] += 1
                res.append(ch)
            else:
                # find the letter(s) with minimal frequency so far
                # and pick the lexicographically smallest among them
                min_freq = min(freq)
                # find the smallest index with freq == min_freq
                for i in range(26):
                    if freq[i] == min_freq:
                        freq[i] += 1
                        res.append(chr(i + ord('a')))
                        break
        
        return "".join(res)