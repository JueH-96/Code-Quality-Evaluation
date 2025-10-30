from typing import List
from collections import defaultdict

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        
        # Step 1: Build a dictionary mapping substrings to the set of indices
        # where they appear in the input array 'arr'.
        # This dictionary will store all unique substrings across all strings,
        # and the original string indices they belong to.
        # Using defaultdict(set) simplifies adding indices to sets.
        all_substring_locations = defaultdict(set)

        # Iterate through each string in the input array
        for i in range(n):
            current_string = arr[i]
            L_i = len(current_string)
            # Generate all substrings of the current string
            # k is the length of the substring
            for k in range(1, L_i + 1):
                # start is the starting index of the substring
                for start in range(L_i - k + 1):
                    sub = current_string[start : start + k]
                    # Add the index of the current string to the set for this substring
                    all_substring_locations[sub].add(i)

        # Step 2: Find the shortest unique substring for each string in arr
        answer = []
        # Iterate through each string in the input array again
        for i in range(n):
            current_string = arr[i]
            L_i = len(current_string)
            
            # Initialize the shortest unique substring for arr[i] to an empty string
            shortest_unique_sub = "" 
            
            # Iterate through substring lengths of the current string, starting from 1
            # We are looking for the *shortest* unique substring, so we iterate
            # by length and stop at the first length where unique substrings exist.
            for k in range(1, L_i + 1):
                unique_subs_of_length_k = []
                # Iterate through all substrings of current_string of the current length k
                for start in range(L_i - k + 1):
                    sub = current_string[start : start + k]
                    
                    # A substring 'sub' is unique to arr[i] if it appears in arr[i]
                    # and does not appear in any other string arr[j] where j != i.
                    # In our all_substring_locations dictionary, this means the set
                    # of indices associated with 'sub' should have size 1.
                    
                    # We know 'sub' comes from arr[i] by how we generate it, so
                    # all_substring_locations[sub] must contain 'i'. Thus, we only
                    # need to check if the size of the set is exactly 1.
                    if len(all_substring_locations[sub]) == 1:
                         unique_subs_of_length_k.append(sub)
                
                # If we found one or more unique substrings of the current length k
                if unique_subs_of_length_k:
                    # Sort them lexicographically to find the smallest one
                    unique_subs_of_length_k.sort()
                    # The first element after sorting is the lexicographically smallest
                    # among all unique substrings of the current length k.
                    shortest_unique_sub = unique_subs_of_length_k[0]
                    
                    # Since we iterate k from 1 upwards, the first unique substring(s)
                    # we find must be the shortest possible ones. We can stop searching
                    # for longer substrings for the current string arr[i].
                    break 

            # Step 3: Store the result for arr[i]
            # If the inner loop finishes without finding any unique substring (i.e.,
            # unique_subs_of_length_k was always empty for all k),
            # shortest_unique_sub remains its initial value "", which is correct.
            answer.append(shortest_unique_sub)

        return answer