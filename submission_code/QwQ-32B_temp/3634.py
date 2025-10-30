import bisect
from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        available = defaultdict(list)
        total = 0

        for i, c in enumerate(s):
            # Compute the mirror character
            mirror_char = chr(ord('a') + 25 - (ord(c) - ord('a')))
            
            # Check if there are any available indices for the mirror character
            if mirror_char in available and available[mirror_char]:
                indices = available[mirror_char]
                pos = bisect.bisect_left(indices, i)
                if pos > 0:
                    j = indices[pos - 1]
                    # Remove the found index from the list
                    indices.pop(pos - 1)
                    total += (i - j)
                    continue  # Do not add current index to available

            # Add current index to the available list for the current character
            available[c].append(i)
        
        return total