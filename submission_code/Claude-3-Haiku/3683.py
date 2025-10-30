from typing import List
from itertools import combinations

class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        # Create a list of all possible splits
        splits = []
        for i in range(1, len(word)):
            splits.append((word[:i], word[i:]))
        
        # Find all unique splits
        unique_splits = set(splits)
        
        # If the number of friends is less than or equal to the number of unique splits, return the lexicographically largest split
        if numFriends <= len(unique_splits):
            return max(unique_splits, key=lambda x: (x[0], x[1]))
        
        # Otherwise, find the lexicographically largest combination of splits
        max_split = None
        for combo in combinations(unique_splits, numFriends):
            combo_str = ''.join(sorted([s[0] for s in combo]) + sorted([s[1] for s in combo]))
            if max_split is None or combo_str > max_split:
                max_split = combo_str
        
        return max_split