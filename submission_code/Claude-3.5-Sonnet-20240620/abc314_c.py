# YOUR CODE HERE
N, M = map(int, input().split())
S = list(input())
C = list(map(int, input().split()))

for i in range(1, M + 1):
    color_indices = [j for j in range(N) if C[j] == i]
    if color_indices:
        chars = [S[j] for j in color_indices]
        chars = [chars[-1]] + chars[:-1]
        for idx, char in zip(color_indices, chars):
            S[idx] = char

print(''.join(S))