class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        base_segments = 0
        start = 0
        while start < n:
            base_segments += 1
            distinct = 0
            freq = [0] * 26
            j = start
            while j < n:
                c_index = ord(s[j]) - ord('a')
                if freq[c_index] == 0:
                    distinct += 1
                freq[c_index] += 1
                if distinct > k:
                    break
                j += 1
            start = j
        ans = base_segments
        for i in range(n):
            for letter in range(26):
                new_char = chr(ord('a') + letter)
                if new_char == s[i]:
                    continue
                segments = 0
                start_idx = 0
                while start_idx < n:
                    segments += 1
                    distinct = 0
                    freq = [0] * 26
                    j = start_idx
                    while j < n:
                        current_char = s[j] if j != i else new_char
                        c_index = ord(current_char) - ord('a')
                        if freq[c_index] == 0:
                            distinct += 1
                        freq[c_index] += 1
                        if distinct > k:
                            break
                        j += 1
                    start_idx = j
                if segments > ans:
                    ans = segments
        return ans