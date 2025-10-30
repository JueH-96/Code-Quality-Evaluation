import collections
from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        n_half = n // 2

        # Precompute prefix sums for character counts
        # prefix_counts[i][j] stores count of char 'a'+j in s[0...i-1]
        prefix_counts = [[0] * 26 for _ in range(n + 1)]
        for i in range(n):
            for j in range(26):
                prefix_counts[i+1][j] = prefix_counts[i][j]
            prefix_counts[i+1][ord(s[i]) - ord('a')] += 1

        # Helper function to get character counts for a range s[start:end] (inclusive)
        def get_range_counts(start: int, end: int) -> collections.Counter:
            counts = collections.Counter()
            for j in range(26):
                counts[chr(ord('a') + j)] = prefix_counts[end+1][j] - prefix_counts[start][j]
            return counts

        results = []

        for query in queries:
            a, b, c, d = query
            possible = True

            # Step 1: Initialize the combined pool of characters from rearrangeable segments.
            rearrangeable_pool = get_range_counts(a, b) + get_range_counts(c, d)

            # Step 2: Iterate through pairs (k, n-1-k) from the first half of the string.
            for k in range(n_half):
                left_idx = k
                right_idx = n - 1 - k

                is_left_rearrangeable = (a <= left_idx <= b)
                is_right_rearrangeable = (c <= right_idx <= d)

                if not is_left_rearrangeable and not is_right_rearrangeable:
                    # Case 1: Both positions are fixed. They must match.
                    if s[left_idx] != s[right_idx]:
                        possible = False
                        break
                elif is_left_rearrangeable and not is_right_rearrangeable:
                    # Case 2: Left is rearrangeable, Right is fixed.
                    # s[left_idx] must become s[right_idx]. Demand s[right_idx] from the pool.
                    rearrangeable_pool[s[right_idx]] -= 1
                elif not is_left_rearrangeable and is_right_rearrangeable:
                    # Case 3: Left is fixed, Right is rearrangeable.
                    # s[right_idx] must become s[left_idx]. Demand s[left_idx] from the pool.
                    rearrangeable_pool[s[left_idx]] -= 1
                else: # is_left_rearrangeable and is_right_rearrangeable
                    # Case 4: Both positions are rearrangeable.
                    # Their original characters are already in the pool. No further demands are made here.
                    # These positions will be filled by pairs from the remaining pool.
                    pass
            
            if not possible:
                results.append(False)
                continue

            # Step 3: Check if all demands were met and remaining characters can form pairs.
            for count in rearrangeable_pool.values():
                if count < 0 or count % 2 != 0:
                    possible = False
                    break
            
            results.append(possible)
        
        return results