N = int(input())
P = list(map(int, input().split()))

max_others = max(P[1:])
if P[0] > max_others:
    print(0)
else:
    print(max_others - P[0] + 1)