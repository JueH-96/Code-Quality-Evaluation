# YOUR CODE HERE
N, D = map(int, input().split())
S = list(input())

for _ in range(D):
    # Find the rightmost '@'
    last_at_index = -1
    for i in range(N-1, -1, -1):
        if S[i] == '@':
            last_at_index = i
            break
    if last_at_index != -1:
        S[last_at_index] = '.'

print(''.join(S))