# Read input values
N = int(input())
bets = []
for _ in range(N):
    C = int(input())
    A = list(map(int, input().split()))
    bets.append((C, A))
X = int(input())

# Find all people who bet on X
candidates = []
for i in range(N):
    C_i, A_i = bets[i]
    if X in A_i:
        candidates.append((C_i, i+1))  # storing (C_i, person number)

if not candidates:
    print(0)
else:
    # Find the minimum C_i among candidates
    min_C = min(candidate[0] for candidate in candidates)
    # Select all candidates with C_i == min_C
    result = [candidate[1] for candidate in candidates if candidate[0] == min_C]
    # Sort the result
    result.sort()
    # Print the count and the result
    print(len(result))
    print(' '.join(map(str, result)))