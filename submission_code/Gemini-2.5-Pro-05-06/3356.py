import collections
from typing import List # For type hinting

class Solution:
  def shortestSubstrings(self, arr: List[str]) -> List[str]:
    n = len(arr)
    # Using collections.defaultdict to store sets of indices for each substring.
    # Key: substring (str), Value: set of indices (int) of strings in 'arr' where the substring appears.
    substring_origins = collections.defaultdict(set)

    # Step 1: Populate the substring_origins map.
    # This map will tell us, for any given substring, which of the original strings in 'arr' contain it.
    # Iterate through each string in the input array 'arr'.
    for i in range(n):
        s = arr[i]
        s_len = len(s)
        # Iterate over all possible substring lengths for the current string 's'.
        # Lengths range from 1 up to the length of 's'.
        for length in range(1, s_len + 1):
            # Iterate over all possible start indices for substrings of the current 'length'.
            # This generates all substrings of 's'.
            for start_idx in range(s_len - length + 1):
                # Extract the substring.
                sub = s[start_idx : start_idx + length]
                # Record that this substring 'sub' is found in the string arr[i].
                # We add 'i' (the index of arr[i]) to the set of origins for 'sub'.
                substring_origins[sub].add(i)
    
    # Step 2: For each string arr[i], find its required shortest unique substring.
    final_answer_list = [] # This list will store the results for each string in 'arr'.
    
    # Iterate through each string in 'arr' again, this time to determine its specific answer.
    for i in range(n):
        s_i = arr[i] # The current string for which we are finding the answer.
        len_si = len(s_i)
        
        # 'current_ans_for_si' will store the result for s_i.
        # It's initialized to "" (empty string), which is the required output if no suitable substring is found.
        current_ans_for_si = "" 

        # Iterate over possible lengths for substrings of s_i.
        # We start with the shortest possible length (1) and go up to len_si.
        # This ensures that the first unique substring we find will be among the shortest ones.
        for length in range(1, len_si + 1):
            # 'candidate_unique_substrings_this_length' will store all substrings of s_i
            # that have the current 'length' and are unique to s_i.
            candidate_unique_substrings_this_length = []
            
            # Generate all substrings of s_i of the current 'length'.
            for start_idx in range(len_si - length + 1):
                sub = s_i[start_idx : start_idx + length]
                
                # Check if this substring 'sub' is unique to s_i.
                # A substring is unique to s_i if it appears in s_i but not in any other arr[j] (where j != i).
                # In terms of our 'substring_origins' map, this means the set of origins for 'sub'
                # must contain only the index 'i'.
                if substring_origins[sub] == {i}: # {i} creates a temporary set for comparison
                    candidate_unique_substrings_this_length.append(sub)
            
            # If any unique substrings of the current 'length' were found:
            if candidate_unique_substrings_this_length:
                # We need the lexicographically smallest among these candidate substrings.
                # Python's min() function on a list of strings achieves this.
                current_ans_for_si = min(candidate_unique_substrings_this_length)
                
                # Since we are iterating 'length' in increasing order, we have now found
                # the shortest unique substring(s). By taking the min, we've also satisfied
                # the lexicographical requirement. No need to check longer substrings.
                # So, we can break from the loop over 'length'.
                break 
        
        # Add the determined answer for s_i (either the found substring or "") to our list of final answers.
        final_answer_list.append(current_ans_for_si)
            
    return final_answer_list