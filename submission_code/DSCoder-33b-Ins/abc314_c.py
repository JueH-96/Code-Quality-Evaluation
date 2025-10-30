# YOUR CODE HERE

N, M = map(int, input().split())
S = list(input())
C = list(map(int, input().split()))

# Create a list of lists to store the indices of each color
color_indices = [[] for _ in range(M)]
for i in range(N):
    color_indices[C[i]-1].append(i)

# Perform the circular shift operation for each color
for indices in color_indices:
    if len(indices) > 1:
        last_element = S[indices[-1]]
        for i in range(len(indices)-1, 0, -1):
            S[indices[i]] = S[indices[i-1]]
        S[indices[0]] = last_element

print(''.join(S))