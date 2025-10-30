import collections
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        """
        Finds the shortest unique substring for each string in an array.

        A substring is unique for arr[i] if it doesn't appear in any other arr[j].
        If multiple shortest unique substrings exist, the lexicographically smallest is chosen.
        """
        
        # Step 1: Pre-computation of substring frequencies.
        # The frequency is the number of strings in 'arr' that contain a given substring.
        substring_counts = collections.defaultdict(int)
        for s in arr:
            # We generate all unique substrings for the string 's'.
            # A set is used to ensure that a substring like "ab" in "abab" is counted only once for this string.
            unique_subs_for_s = set()
            for i in range(len(s)):
                for j in range(i, len(s)):
                    unique_subs_for_s.add(s[i:j+1])
            
            # For each unique substring, we increment its overall count.
            for sub in unique_subs_for_s:
                substring_counts[sub] += 1
        
        # Step 2: Find the required substring for each string in 'arr'.
        answer = []
        for current_string in arr:
            # Initialize the best found substring for the current_string as empty.
            best_sub = ""
            
            # We need the shortest, then lexicographically smallest.
            # Iterating by length from 1 upwards ensures we find the shortest first.
            for length in range(1, len(current_string) + 1):
                
                # Collect all valid substrings of the current length.
                candidates = []
                for i in range(len(current_string) - length + 1):
                    sub = current_string[i : i + length]
                    
                    # A substring is unique to 'current_string' if its total count is 1.
                    if substring_counts[sub] == 1:
                        candidates.append(sub)
                
                # If we have found candidates at this length, they are the shortest possible.
                if candidates:
                    # From the shortest candidates, we pick the lexicographically smallest.
                    best_sub = min(candidates)
                    # We have found the answer for 'current_string', so we can stop searching.
                    break
            
            # Add the found substring (or "" if none was found) to our answer list.
            answer.append(best_sub)
            
        return answer