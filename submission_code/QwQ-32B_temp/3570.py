import bisect
from collections import defaultdict

class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        
        # Step 1: Compute prefix consonant counts
        prefix_c = [0] * (n + 1)
        for i in range(1, n + 1):
            if word[i-1] in vowels:
                prefix_c[i] = prefix_c[i-1]
            else:
                prefix_c[i] = prefix_c[i-1] + 1
        
        # Step 2: Compute last_occurrence and min_last arrays
        last_occurrence = {'a': -1, 'e': -1, 'i': -1, 'o': -1, 'u': -1}
        min_last = [0] * n
        for j in range(n):
            char = word[j]
            if char in vowels:
                last_occurrence[char] = j
            current_min = min(last_occurrence.values())
            min_last[j] = current_min
        
        # Step 3: Build prefix_indices dictionary
        prefix_indices = defaultdict(list)
        for i in range(n + 1):
            pc = prefix_c[i]
            prefix_indices[pc].append(i)
        
        result = 0
        
        # Step 4: Iterate over each possible end position j
        for j in range(n):
            target = prefix_c[j + 1] - k
            if target not in prefix_indices:
                continue
            list_i = prefix_indices[target]
            min_last_j = min_last[j]
            
            if min_last_j < 0:
                continue  # Not all vowels present up to j
            
            # Find all i in list_i where i <= j and i <= min_last_j
            # First, find indices <= j
            idx_j = bisect.bisect_right(list_i, j)
            # Now, within those, find indices <= min_last_j
            idx_min = bisect.bisect_right(list_i, min_last_j, 0, idx_j)
            result += idx_min
        
        return result