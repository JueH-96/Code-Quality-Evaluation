import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:

        def simple_overlap_append(s1, s2):
            """
            Appends s2 to s1 with maximum possible overlap.
            Returns the resulting string s1[:len(s1)-k] + s2, where s1[-k:] == s2[:k]
            and k is maximized.
            This resulting string contains s1 and s2 as substrings.
            """
            max_overlap = 0
            min_len = min(len(s1), len(s2))
            # Check overlaps from min_len down to 1
            for k in range(min_len, 0, -1):
                # Check if the suffix of s1 of length k matches the prefix of s2 of length k
                if s1[len(s1) - k:] == s2[:k]:
                    max_overlap = k
                    break # Found the largest overlap

            # The resulting string is the part of s1 before the overlap + the full s2
            # If max_overlap is 0, it means no positive overlap was found, k=0.
            # s1[:len(s1)-0] + s2 = s1 + s2.
            return s1[:len(s1) - max_overlap] + s2

        # 1. Filter strings: keep only those that are not a substring of others
        # among the unique input strings.
        original_strings = list(set([a, b, c]))

        essential_strings = []
        for s1 in original_strings:
            is_essential = True
            for s2 in original_strings:
                # Check if s1 is a substring of a *different* string s2 in the unique list
                if s1 != s2 and s1 in s2:
                    is_essential = False
                    break
            if is_essential:
                essential_strings.append(s1)

        n = len(essential_strings)
        best_res = ""

        # The problem guarantees 1 <= length of inputs, so n >= 1 after filtering

        if n == 1:
            # If only one essential string remains, it must contain the others
            best_res = essential_strings[0]
        elif n == 2:
            # If two essential strings remain, try merging in both orders
            s1, s2 = essential_strings
            res1 = simple_overlap_append(s1, s2)
            res2 = simple_overlap_append(s2, s1)

            # Choose the shorter result, then the lexicographically smaller one
            if len(res1) < len(res2):
                best_res = res1
            elif len(res2) < len(res1):
                best_res = res2
            else:
                best_res = min(res1, res2)
        elif n == 3:
            # If three essential strings remain, try all 6 permutations
            s1, s2, s3 = essential_strings
            perms = list(itertools.permutations([s1, s2, s3]))

            # Initialize best_res with the result of the first permutation
            p = perms[0]
            # Merge the first two strings, then merge the result with the third
            best_res = simple_overlap_append(simple_overlap_append(p[0], p[1]), p[2])

            # Iterate through the rest of the permutations
            for p1, p2, p3 in perms[1:]:
                # Merge the first two strings in the current permutation
                temp_res = simple_overlap_append(p1, p2)
                # Merge the result with the third string
                current_res = simple_overlap_append(temp_res, p3)

                # Update best_res based on length and lexicographical order
                if len(current_res) < len(best_res):
                    best_res = current_res
                elif len(current_res) == len(best_res):
                    best_res = min(best_res, current_res)

        return best_res