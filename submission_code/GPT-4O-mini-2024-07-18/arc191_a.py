def maximize_string(N, M, S, T):
    # Convert S to a list for mutability
    S = list(S)
    
    # Sort T in descending order to maximize the replacements
    T_sorted = sorted(T, reverse=True)
    
    # Perform the replacements
    for k in range(M):
        # We are replacing the k-th character of T_sorted
        # Find the best position in S to replace
        for i in range(N):
            if T_sorted[k] > S[i]:
                S[i] = T_sorted[k]
                break
    
    # Join the list back to a string and return as an integer
    return int(''.join(S))

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

# First line contains N and M
N, M = map(int, data[0].split())
# Second line contains S
S = data[1].strip()
# Third line contains T
T = data[2].strip()

# Get the result
result = maximize_string(N, M, S, T)

# Print the result
print(result)