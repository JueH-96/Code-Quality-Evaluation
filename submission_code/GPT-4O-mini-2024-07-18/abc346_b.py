def can_form_substring(W, B):
    # The pattern we are working with
    pattern = "wbwbwwbwbwbw"
    pattern_length = len(pattern)

    # We need to check if we can form W 'w's and B 'b's
    # We can create a substring of the pattern repeated enough times
    # to cover the maximum needed 'w's and 'b's.
    
    # Create a long enough string by repeating the pattern
    # We repeat it enough times to ensure we can check all combinations
    extended_pattern = pattern * ((W + B) // pattern_length + 2)

    # Check all possible substrings of the extended pattern
    for start in range(len(extended_pattern)):
        w_count = 0
        b_count = 0
        for end in range(start, len(extended_pattern)):
            if extended_pattern[end] == 'w':
                w_count += 1
            else:
                b_count += 1
            
            # If we have found enough 'w's and 'b's, we can return "Yes"
            if w_count == W and b_count == B:
                return "Yes"
            # If we exceed the required counts, we can break early
            if w_count > W or b_count > B:
                break

    return "No"

import sys
input = sys.stdin.read
W, B = map(int, input().strip().split())
print(can_form_substring(W, B))