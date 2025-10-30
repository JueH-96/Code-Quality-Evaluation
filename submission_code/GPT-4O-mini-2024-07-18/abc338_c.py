def max_servings(N, Q, A, B):
    max_servings = 0
    
    # Iterate over all possible servings of dish A
    for servings_A in range(0, min(Q[i] // A[i] if A[i] > 0 else float('inf') for i in range(N)) + 1):
        # Calculate remaining ingredients after making servings_A of dish A
        remaining = [Q[i] - servings_A * A[i] for i in range(N)]
        
        # Calculate the maximum servings of dish B we can make with the remaining ingredients
        servings_B = float('inf')
        for i in range(N):
            if B[i] > 0:
                servings_B = min(servings_B, remaining[i] // B[i])
        
        # Update the maximum total servings
        max_servings = max(max_servings, servings_A + servings_B)
    
    return max_servings

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
Q = list(map(int, data[1].split()))
A = list(map(int, data[2].split()))
B = list(map(int, data[3].split()))

result = max_servings(N, Q, A, B)
print(result)