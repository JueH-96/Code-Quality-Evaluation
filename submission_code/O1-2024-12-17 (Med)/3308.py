class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # 1) Count frequency of each character
        freq = [0] * 26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1

        # 2) Find the maximum frequency among all characters
        max_freq = max(freq)

        # 3) We need to remove the first (max_freq - 1) occurrences of each character
        #    and keep the rest (this simulates the string right before the last operation)
        removed = [0] * 26
        result = []

        for ch in s:
            idx = ord(ch) - ord('a')
            if removed[idx] < max_freq - 1:
                removed[idx] += 1
            else:
                result.append(ch)

        return "".join(result)