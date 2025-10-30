import sys

n,k,x = map(int,input().split())
t=[0]*n+[-1]*(k-1)
s=list(map(int, sys.stdin.readline().split()))
t[:n]=s
total=t[n-1]-t[0]

for i in range(n+1,n+k):
    t[i]=t[i-k]+x
for i in range(n-k+1,n+1):
    total+=t[i]-t[i-k]
n+=k-1

for i in range(k):
    key=t[i]
    j=min(n-1, t[i+(k-1)//2]+x)
    while j>i:
        if key<t[j-(k-1)//2]:
            t[j]=t[j-(k-1)//2]
            j-=1
        else: break
    t[j]=key

for i in range(n):
    if i<k:
        total-=t[i]
    else:
        total-=t[i]
        total+=t[i-k+1]

    if i<k-1:
        pass
    elif i%k==k-1:
        total+=x
    else:
        if t[i]<t[i-(k-1)//2]+x:
            total+=t[i]-t[i-(k-1)//2]
print(total)