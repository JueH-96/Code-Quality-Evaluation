# YOUR CODE HERE
def can_form_substring(W, B):
    # The repeating pattern
    pattern = "wbwbwwbwbwbw"
    
    # We need to check if we can find W 'w's and B 'b's in a substring of the pattern
    # Since the pattern is 12 characters long, we can check up to 12*2 characters to be sure
    extended_pattern = pattern * 2
    
    # We will use a sliding window approach to check all substrings of length W+B
    for i in range(len(extended_pattern) - (W + B) + 1):
        substring = extended_pattern[i:i + W + B]
        if substring.count('w') == W and substring.count('b') == B:
            return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
W, B = map(int, input().strip().split())
print(can_form_substring(W, B))