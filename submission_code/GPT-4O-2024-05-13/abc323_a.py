# YOUR CODE HERE
S = input().strip()

def check_even_positions(S):
    for i in range(1, 16, 2):
        if S[i] != '0':
            return "No"
    return "Yes"

print(check_even_positions(S))