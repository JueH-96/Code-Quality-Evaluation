N = int(input())
P = list(map(int, input().split()))

if N == 1:
    print(0)
else:
    max_p = max(P[1:])
    x = max(0, max_p - P[0] + 1)
    print(x)