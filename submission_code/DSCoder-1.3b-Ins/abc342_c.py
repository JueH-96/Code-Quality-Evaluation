N = int(input())
S = input()
Q = int(input())

for _ in range(Q):
    c, d = input().split()
    S = S.replace(c, d)

print(S)