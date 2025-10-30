# YOUR CODE HERE
import sys

def is_airport_code(S, T):
    if len(T) != 3:
        return "No"
    
    # Check if T can be formed by a subsequence of length 3 from S
    def can_form_by_3(S, T):
        it = iter(S)
        return all(char in it for char in T.lower())
    
    # Check if T can be formed by a subsequence of length 2 from S and appending 'X'
    def can_form_by_2_and_X(S, T):
        if T[2] != 'X':
            return False
        it = iter(S)
        return all(char in it for char in T[:2].lower())
    
    if can_form_by_3(S, T) or can_form_by_2_and_X(S, T):
        return "Yes"
    else:
        return "No"

input = sys.stdin.read().strip().split()
S = input[0]
T = input[1]

print(is_airport_code(S, T))