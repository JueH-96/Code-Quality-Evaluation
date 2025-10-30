N, M = map(int, input().split())
S = input()
T = input()

def is_possible(S, T, M):
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            return True
    return False

print("Yes" if is_possible(S, T, M) else "No")