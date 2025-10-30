N,L,R = map(int,input().split())
A = list(map(int,input().split()))

ans = []
for a in A:
    if a <= L:
        ans.append(L)
    elif a >= R:
        ans.append(R)
    else:
        if a-L <= R-a:
            ans.append(L)
        else:
            ans.append(R)
print(*ans)