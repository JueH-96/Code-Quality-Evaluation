# YOUR CODE HERE
def are_similar_strings(N, S, T):
    for i in range(N):
        if S[i] == T[i]:
            continue
        if (S[i] == '1' and T[i] == 'l') or (S[i] == 'l' and T[i] == '1'):
            continue
        if (S[i] == '0' and T[i] == 'o') or (S[i] == 'o' and T[i] == '0'):
            continue
        return "No"
    return "Yes"

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]
T = data[2]

# Output result
print(are_similar_strings(N, S, T))