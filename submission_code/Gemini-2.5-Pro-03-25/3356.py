import collections
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        """
        Finds the shortest unique substring for each string in the input array.

        A substring is unique for arr[i] if it doesn't appear as a substring in any other 
        string arr[j] where j != i.
        If multiple shortest unique substrings exist for arr[i], the lexicographically 
        smallest one is chosen.
        If no unique substring exists for arr[i], the result for that index is an empty string.

        Args:
            arr: A list of non-empty strings consisting of lowercase English letters.

        Returns:
            A list of strings, where the i-th string is the shortest unique substring
            for arr[i] (or the lexicographically smallest among shortest), or an 
            empty string if no such substring exists.
        """
        n = len(arr)
        
        # Step 1: Precompute all substrings and the indices of the strings they appear in.
        # Use collections.defaultdict(set) for efficient storage.
        # Key: substring (str)
        # Value: set of indices (int) where the substring appears in arr.
        substring_map = collections.defaultdict(set)

        for i in range(n):
            s = arr[i]
            s_len = len(s)
            # Generate all substrings of the current string s = arr[i]
            for length in range(1, s_len + 1):
                for start in range(s_len - length + 1):
                    # Extract the substring
                    sub = s[start : start + length]
                    # Record that this substring 'sub' is present in the string at index 'i'
                    # The set automatically handles duplicates if 'sub' appears multiple times in s.
                    substring_map[sub].add(i)

        # Step 2: For each string arr[i], find its shortest unique substring.
        answer = [""] * n  # Initialize the result array with empty strings

        for i in range(n):
            current_string = arr[i]
            s_len = len(current_string)
            
            # Variables to keep track of the best unique substring found so far for arr[i].
            # We iterate through lengths, so the first length where we find unique substrings
            # guarantees the shortest length.
            best_sub_for_i = "" 
            # We don't explicitly need min_len_found because the outer loop handles length.

            # Iterate through possible lengths of substrings, from shortest (1) to longest (s_len).
            for length in range(1, s_len + 1):
                
                # Track the best candidate (lexicographically smallest) for the current length.
                current_best_this_length = ""
                found_candidate_this_length = False

                # Iterate through all substrings of arr[i] with the current length.
                for start in range(s_len - length + 1):
                    sub = current_string[start : start + length]

                    # Check if this substring is unique to arr[i] using the precomputed map.
                    # A substring 'sub' generated from arr[i] is unique to arr[i] if and only if
                    # it appears in exactly one string in the original array 'arr'.
                    # Since we know 'sub' appears in arr[i] (as we generated it from there),
                    # uniqueness means the set of indices for 'sub' in substring_map has size 1.
                    if len(substring_map[sub]) == 1: 
                        # This substring 'sub' is unique to arr[i].
                        
                        if not found_candidate_this_length:
                            # This is the first unique substring of this length we've found.
                            current_best_this_length = sub
                            found_candidate_this_length = True
                        elif sub < current_best_this_length:
                            # We found another unique substring of the same length,
                            # but this one is lexicographically smaller. Update the best for this length.
                            current_best_this_length = sub

                # After checking all substrings of the current length 'length':
                # If we found any unique substring(s) of this length:
                if found_candidate_this_length:
                    # This 'length' is the shortest possible length for a unique substring for arr[i].
                    # 'current_best_this_length' now holds the lexicographically smallest unique substring
                    # of this shortest length.
                    answer[i] = current_best_this_length
                    # We've found the answer for arr[i], so we can stop searching longer lengths.
                    break 
            
            # If the outer 'length' loop completes without the 'break' being hit, 
            # it means no unique substring was found for arr[i]. 
            # In this case, answer[i] remains the initialized empty string "".

        return answer