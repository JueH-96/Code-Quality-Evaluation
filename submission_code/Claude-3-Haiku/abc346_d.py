import sys

# Read the input
N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()
C = [int(x) for x in sys.stdin.readline().split()]

# Initialize the minimum cost to 0
min_cost = 0

# Iterate through the string S
for i in range(N-1):
    # If the current and next characters are the same, continue
    if S[i] == S[i+1]:
        continue
    
    # Find the cheaper operation to make the characters different
    if C[i] <= C[i+1]:
        min_cost += C[i]
        S = S[:i] + ('1' if S[i] == '0' else '0') + S[i+1:]
    else:
        min_cost += C[i+1]
        S = S[:i+1] + ('1' if S[i+1] == '0' else '0') + S[i+2:]

print(min_cost)