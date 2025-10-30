def is_airport_code(S, T):
    # Check if T can be formed by a subsequence of length 3
    def can_form_subsequence_of_length_3(S, T):
        it = iter(S)
        return all(char in it for char in T)

    # Check if T can be formed by a subsequence of length 2 and appending 'X'
    def can_form_subsequence_of_length_2(S, T):
        if T[2] != 'X':
            return False
        # Check for the first two characters of T
        first_two = T[:2]
        it = iter(S)
        return all(char in it for char in first_two)

    # Check both conditions
    if can_form_subsequence_of_length_3(S, T) or can_form_subsequence_of_length_2(S, T):
        return "Yes"
    else:
        return "No"

import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
T = data[1]

print(is_airport_code(S, T))