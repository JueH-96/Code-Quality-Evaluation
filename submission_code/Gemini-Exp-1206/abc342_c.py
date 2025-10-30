N = int(input())
S = list(input())
Q = int(input())
for _ in range(Q):
    c, d = input().split()
    for i in range(N):
        if S[i] == c:
            S[i] = d
print("".join(S))