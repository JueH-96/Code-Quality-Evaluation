# YOUR CODE HERE
N = int(input())
S = input().strip()
C = list(map(int, input().split()))

min_cost = float('inf')

# Try making each position i the one with adjacent equal characters
for i in range(N - 1):
    cost = 0
    
    # For position i, we want S[i] == S[i+1]
    if S[i] != S[i + 1]:
        # They are different, so we need to flip one of them
        cost += min(C[i], C[i + 1])
    
    # For all other positions j, we want S[j] != S[j+1]
    for j in range(N - 1):
        if j != i and S[j] == S[j + 1]:
            # They are the same, so we need to flip one of them
            cost += min(C[j], C[j + 1])
    
    min_cost = min(min_cost, cost)

print(min_cost)