import collections

class Solution:
    def maximumLength(self, s: str) -> int:
        # char_to_len_counts[char][length] will store the total count
        # of the special substring (char * length) in s.
        # This uses defaultdict for convenient initialization of nested dictionaries.
        char_to_len_counts = collections.defaultdict(lambda: collections.defaultdict(int))

        n = len(s)
        i = 0
        while i < n:
            current_char = s[i]
            j = i
            # Find the end of the current consecutive run of `current_char`
            while j < n and s[j] == current_char:
                j += 1
            
            run_length = j - i
            
            # For this run, calculate how many times each special substring of `current_char`
            # can be formed.
            # A special substring of length `L` (i.e., `current_char * L`)
            # appears `(run_length - L + 1)` times within this specific run.
            # These counts are accumulated into `char_to_len_counts`.
            for length in range(1, run_length + 1):
                char_to_len_counts[current_char][length] += (run_length - length + 1)
            
            # Move `i` to the position after the current run to find the next one
            i = j

        max_found_length = -1

        # Iterate through the accumulated counts to find the maximum length
        # of a special substring that occurs at least thrice.
        for char_code in char_to_len_counts:
            for length, count in char_to_len_counts[char_code].items():
                if count >= 3:
                    max_found_length = max(max_found_length, length)
        
        return max_found_length