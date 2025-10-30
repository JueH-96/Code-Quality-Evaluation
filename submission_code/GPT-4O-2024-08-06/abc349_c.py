# YOUR CODE HERE
def is_airport_code(S, T):
    # Check if T can be formed by a subsequence of length 3 from S
    def can_form_subsequence_of_length_3(S, T):
        t_index = 0
        for char in S:
            if char.upper() == T[t_index]:
                t_index += 1
                if t_index == 3:
                    return True
        return False

    # Check if T can be formed by a subsequence of length 2 from S and appending 'X'
    def can_form_subsequence_of_length_2_with_X(S, T):
        t_index = 0
        for char in S:
            if char.upper() == T[t_index]:
                t_index += 1
                if t_index == 2:
                    return T[2] == 'X'
        return False

    # Check both conditions
    if can_form_subsequence_of_length_3(S, T) or can_form_subsequence_of_length_2_with_X(S, T):
        return "Yes"
    else:
        return "No"

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
T = data[1]

# Output result
print(is_airport_code(S, T))