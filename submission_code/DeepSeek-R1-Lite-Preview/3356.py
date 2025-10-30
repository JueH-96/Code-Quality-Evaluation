from typing import List
from collections import defaultdict

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        # Generate all unique substrings for each string
        subsets = [set(arr[i][j:k] for j in range(len(arr[i])) for k in range(j+1, len(arr[i])+1)) for i in range(len(arr))]
        
        # Build frequency dictionary: substring -> count of strings containing it
        freq = defaultdict(int)
        for substr_set in subsets:
            for substr in substr_set:
                freq[substr] += 1
        
        # For each string, find the shortest substring not present in any other string
        answer = []
        for i in range(len(arr)):
            unique_subs = [substr for substr in subsets[i] if freq[substr] == 1]
            if unique_subs:
                # Find the minimum length among unique substrings
                min_length = min(len(substr) for substr in unique_subs)
                # Filter substrings with the minimum length
                candidates = [substr for substr in unique_subs if len(substr) == min_length]
                # Choose the lexicographically smallest substring
                answer.append(min(candidates))
            else:
                answer.append("")
        
        return answer