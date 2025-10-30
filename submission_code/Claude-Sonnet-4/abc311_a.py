# YOUR CODE HERE
N = int(input())
S = input()

seen = set()
for i in range(N):
    seen.add(S[i])
    if len(seen) == 3:  # All of A, B, C have appeared
        print(i + 1)  # +1 because we want 1-indexed position
        break