from typing import defaultdict
from collections import deque

class Solution:
    def calculateScore(self, s: str) -> int:
        # Precompute the mirror for each letter
        mirror_dict = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            mirror_position = 25 - (ord(c) - ord('a'))
            mirror_letter = chr(ord('a') + mirror_position)
            mirror_dict[c] = mirror_letter
        
        # Create a dictionary to hold dequeues of positions for each letter
        position_dict = defaultdict(deque)
        for i, c in enumerate(s):
            position_dict[c].append(i)
        
        score = 0
        # Iterate through the string from left to right
        for i, c in enumerate(s):
            m = mirror_dict[c]
            if m in position_dict:
                # Find the closest j to the left of i where s[j] == m
                while position_dict[m]:
                    j = position_dict[m][0]
                    if j < i:
                        # Found a valid j
                        position_dict[m].popleft()
                        score += i - j
                        break
                    else:
                        # j is not to the left of i, remove it
                        position_dict[m].popleft()
                else:
                    # No j to the left of i for this mirror letter
                    continue
            else:
                continue
        return score