class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        freq = [0] * 26
        last_index = [-1] * 26
        for i, char in enumerate(s):
            idx = ord(char) - ord('a')
            freq[idx] += 1
            last_index[idx] = i
        M = max(freq)
        positions = []
        for idx in range(26):
            if freq[idx] == M:
                positions.append(last_index[idx])
        positions.sort()
        result_str = ''.join(s[pos] for pos in positions)
        return result_str