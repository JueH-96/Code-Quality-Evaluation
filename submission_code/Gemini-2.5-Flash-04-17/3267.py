import collections

class Solution:
    def maximumLength(self, s: str) -> int:
        # Step 1: Parse the string to find consecutive runs and their lengths
        # char_runs[c] will be a list of lengths of consecutive runs of character c
        char_runs = collections.defaultdict(list)
        n = len(s)
        # Constraint 3 <= n <= 50, so n is never 0 or < 3

        current_char = s[0]
        current_len = 1
        for i in range(1, n):
            if s[i] == current_char:
                current_len += 1
            else:
                char_runs[current_char].append(current_len)
                current_char = s[i]
                current_len = 1
        # Append the last run after the loop finishes
        char_runs[current_char].append(current_len)

        # Step 3 & 4: Calculate total counts for each character and length
        # char_counts[c][k] stores the total occurrences of special substring c * k
        # We use nested defaultdict to handle missing characters or lengths gracefully
        char_counts = collections.defaultdict(lambda: collections.defaultdict(int))

        # For each character and its runs, calculate how many times each special substring length appears
        for char, run_list in char_runs.items():
            for li in run_list:
                # A run of length li contributes (li - k + 1) occurrences of c * k
                # for each length k from 1 up to li.
                for k in range(1, li + 1):
                    char_counts[char][k] += (li - k + 1)

        # Step 5, 6: Iterate through possible lengths L from the maximum possible (n) down to 1
        # We want the *longest* length, so the first length we find (iterating downwards)
        # that satisfies the condition (occurs at least thrice) is our answer.
        for length_to_check in range(n, 0, -1):
            # Check this length for all characters present in the string
            # Iterating keys of char_runs is sufficient as char_counts is populated
            # only for characters present in char_runs.
            for char in char_runs: 
                # Use the pre-calculated count. If length_to_check is not in char_counts[char],
                # the defaultdict for int returns 0, which is correct.
                if char_counts[char][length_to_check] >= 3:
                    # Found a special substring of length length_to_check that occurs >= 3 times
                    return length_to_check

        # Step 7: If the loops complete without finding such a length, return -1
        return -1