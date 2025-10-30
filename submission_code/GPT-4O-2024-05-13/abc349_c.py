# YOUR CODE HERE
def is_airport_code(S, T):
    # Check if T can be formed by a subsequence of length 3 from S
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if S[i].upper() == T[0] and S[j].upper() == T[1] and S[k].upper() == T[2]:
                    return "Yes"
    
    # Check if T can be formed by a subsequence of length 2 from S and appending 'X'
    for i in range(n):
        for j in range(i + 1, n):
            if S[i].upper() == T[0] and S[j].upper() == T[1] and T[2] == 'X':
                return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
data = input().split()
S = data[0]
T = data[1]

print(is_airport_code(S, T))