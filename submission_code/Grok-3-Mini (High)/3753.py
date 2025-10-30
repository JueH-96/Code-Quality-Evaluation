class Solution:
    def maxDifference(self, s: str) -> int:
        freq = [0] * 26
        for char in s:
            index = ord(char) - ord('a')
            freq[index] += 1
        max_odd = -1
        min_even = 101
        for count in freq:
            if count > 0:
                if count % 2 == 1 and count > max_odd:
                    max_odd = count
                elif count % 2 == 0 and count < min_even:
                    min_even = count
        return max_odd - min_even