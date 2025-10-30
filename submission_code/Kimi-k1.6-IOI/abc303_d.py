X, Y, Z = map(int, input().split())
S = input().strip()

INF = 1 << 60
prev0, prev1 = 0, INF

for c in S:
    if c == 'a':
        cost0 = X  # State 0: press 'a' alone
        cost1 = Y  # State 1: press Shift+'a'
    else:
        cost0 = Y  # State 0: press Shift+'a'
        cost1 = X  # State 1: press 'a' alone
    
    curr0 = min(prev0 + cost0, prev1 + Z + cost0)
    curr1 = min(prev0 + Z + cost1, prev1 + cost1)
    prev0, prev1 = curr0, curr1

print(min(prev0, prev1))