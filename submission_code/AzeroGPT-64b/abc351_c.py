import sys

input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))

ans = 0
buff = 0
stk = []

for a in A:
    a = 2**a
    if not stk:
        stk.append(a)
        buff = 0
        ans += 1
    else:
        if stk[-1] == a:
            buff += 1
            if buff % 2 == 1:
                stk[-1] *= 2
            else:
                stk.pop()
        else:
            stk.append(a)
            buff = 0
            ans += 1
                
print(ans)