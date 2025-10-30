# YOUR CODE HERE
N, M = map(int, input().split())
S = input()
T = input()

def solve(N, M, S, T):
    X = ['#'] * N
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            X[i:i+M] = list(T)
    return ''.join(X) == S

print('Yes' if solve(N, M, S, T) else 'No')