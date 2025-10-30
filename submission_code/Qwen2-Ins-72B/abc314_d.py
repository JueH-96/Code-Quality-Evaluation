import sys
input = sys.stdin.readline

N = int(input())
S = list(input().strip())
Q = int(input())

ops = [list(input().split()) for _ in range(Q)]

lower = False
upper = False
change = {}

for i in range(Q-1, -1, -1):
    op = ops[i]
    if op[0] == '1':
        if lower:
            change[int(op[1])-1] = op[2].lower()
        elif upper:
            change[int(op[1])-1] = op[2].upper()
        else:
            change[int(op[1])-1] = op[2]
    elif op[0] == '2':
        lower = True
    elif op[0] == '3':
        upper = True

ans = []
for i, s in enumerate(S):
    if i in change:
        ans.append(change[i])
    elif lower:
        ans.append(s.lower())
    elif upper:
        ans.append(s.upper())
    else:
        ans.append(s)

print(''.join(ans))