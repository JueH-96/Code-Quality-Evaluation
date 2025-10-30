class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        if k == 0:
            return True

        # Precompute first and last occurrences for each character
        first_occurrence = {}
        last_occurrence = {}
        for i, char in enumerate(s):
            if char not in first_occurrence:
                first_occurrence[char] = i
            last_occurrence[char] = i

        # List to store the special substring ranges [start_index, end_index]
        # A range [i, j] represents the substring s[i : j + 1]
        special_ranges = []

        # Iterate through all possible start indices i
        for i in range(n):
            # Initialize min/max occurrence indices for the current substring s[i...j]
            # min_f_so_far: Minimum first occurrence index among characters encountered in s[i...j]
            min_f_so_far = n 
            # max_l_so_far: Maximum last occurrence index among characters encountered in s[i...j]
            max_l_so_far = -1

            # Iterate through all possible end indices j (where j >= i)
            for j in range(i, n):
                char = s[j]
                
                # Update min/max occurrence indices as we extend the substring to include s[j]
                min_f_so_far = min(min_f_so_far, first_occurrence[char])
                max_l_so_far = max(max_l_so_far, last_occurrence[char])

                # Optimization: If the minimum first occurrence found among characters in s[i...j]
                # is less than the starting index i, then the current substring s[i:j+1]
                # is not special because at least one character inside it appears before index i.
                # Furthermore, any longer substring starting at i and including s[j] will also
                # contain this character, so it also cannot be special. We can stop checking
                # substrings starting at i that extend to or past j.
                if min_f_so_far < i:
                    break # Break the inner loop (over j) for this starting index i

                # If we reached here, it means all characters encountered in s[i...j] so far
                # have their first occurrence index greater than or equal to i (`min_f_so_far >= i`).
                # For the substring s[i:j+1] to be special, two conditions must be met:
                # 1. All characters present in s[i:j+1] must have their first occurrence >= i.
                #    This is true if and only if the minimum first occurrence among these characters is exactly i.
                # 2. All characters present in s[i:j+1] must have their last occurrence <= j.
                #    This is true if and only if the maximum last occurrence among these characters is exactly j.
                #
                # The condition `min_f_so_far == i` and `max_l_so_far == j` together signify that
                # the set of characters whose entire occurrence range [first, last] falls
                # exactly within [i, j] is precisely the set of characters present in s[i:j+1].
                # This is the defining property of a special substring s[i:j+1].

                if min_f_so_far == i and max_l_so_far == j:
                    # Found a candidate special range [i, j].
                    # Check if the corresponding substring s[i:j+1] is not the entire string s.
                    # The length is j - i + 1. The length of s is n.
                    if j - i + 1 < n:
                        special_ranges.append((i, j))

        # Now we have a list of all special substring ranges [start, end].
        # We need to find the maximum number of disjoint intervals from this list.
        # This is a standard greedy problem solved by sorting intervals by their end points.

        # Sort the special ranges based on their end coordinate
        special_ranges.sort(key=lambda x: x[1])

        # Count the maximum number of mutually disjoint intervals
        count = 0
        # Initialize the end coordinate of the last selected interval to -1,
        # which is smaller than any possible index (>= 0).
        last_end = -1

        for start, end in special_ranges:
            # If the current interval starts strictly after the previous selected interval ends,
            # they are disjoint. Select the current interval.
            if start > last_end:
                count += 1
                # Update the end coordinate of the last selected interval
                last_end = end

        # The maximum number of disjoint special substrings is 'count'.
        # Check if it is possible to select at least k such substrings.
        return count >= k