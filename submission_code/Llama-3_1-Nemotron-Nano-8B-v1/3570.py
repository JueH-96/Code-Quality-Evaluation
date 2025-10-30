import bisect
from collections import defaultdict

class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        if n < 5:
            return 0
        
        # Precompute prefix consonant array
        prefix_consonant = [0] * (n + 1)
        for i in range(n):
            prefix_consonant[i+1] = prefix_consonant[i] + (0 if word[i] in vowels else 1)
        
        # Precompute value to indices mapping
        value_to_indices = defaultdict(list)
        for idx, val in enumerate(prefix_consonant):
            value_to_indices[val].append(idx)
        
        count = 0
        
        for i in range(n):
            vowels_present = set()
            j_min = None
            for j in range(i, n):
                c = word[j]
                if c in vowels:
                    vowels_present.add(c)
                if len(vowels_present) == 5:
                    j_min = j
                    break
            if j_min is None:
                continue  # No valid substring starting at i
            
            target = prefix_consonant[i] + k
            if target not in value_to_indices:
                continue
            list_x = value_to_indices[target]
            # Find the first index in list_x >= j_min + 1
            pos = bisect.bisect_left(list_x, j_min + 1)
            count += len(list_x) - pos
        
        return count