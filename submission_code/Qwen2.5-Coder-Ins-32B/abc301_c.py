# YOUR CODE HERE
import sys
from collections import Counter

def can_win(S, T):
    # Count the occurrences of each character in S and T
    count_S = Counter(S)
    count_T = Counter(T)
    
    # Count the number of @ symbols in S and T
    at_S = count_S['@']
    at_T = count_T['@']
    
    # Remove the @ symbols from the counters
    del count_S['@']
    del count_T['@']
    
    # Check if the remaining characters can be rearranged to match
    for char in count_S:
        if count_S[char] > count_T[char]:
            if count_S[char] - count_T[char] > at_T:
                return "No"
            at_T -= count_S[char] - count_T[char]
        elif count_T[char] > count_S[char]:
            if count_T[char] - count_S[char] > at_S:
                return "No"
            at_S -= count_T[char] - count_S[char]
    
    return "Yes"

# Read input
input = sys.stdin.read().strip().split()
S = input[0]
T = input[1]

# Output the result
print(can_win(S, T))