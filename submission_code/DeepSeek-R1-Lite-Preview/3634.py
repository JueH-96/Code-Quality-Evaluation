from bisect import bisect_right
from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        # Function to get the mirror of a character
        def get_mirror(c):
            return chr(ord('a') + 25 - (ord(c) - ord('a')))
        
        # Create a dictionary to hold lists of positions for each character
        pos = defaultdict(list)
        for idx, char in enumerate(s):
            pos[char].append(idx)
        
        score = 0
        
        # Iterate through each position in the string
        for i, char in enumerate(s):
            m = get_mirror(char)
            if m in pos:
                m_list = pos[m]
                # Find the insertion point for i in m_list
                insertion_point = bisect_right(m_list, i)
                if insertion_point > 0:
                    j = m_list[insertion_point - 1]
                    score += i - j
                    # Remove j from m_list
                    m_list.pop(insertion_point - 1)
        
        return score