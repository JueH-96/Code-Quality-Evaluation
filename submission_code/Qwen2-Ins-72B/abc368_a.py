n,k=map(int,input().split())
a=list(map(int,input().split()))
b=a[k:]+a[:k]
print(*b)