n,a,b=map(int,input().split())
d=list(map(int,input().split()))
maxd=b+a-max(d)+1
if maxd<=a:
    print('Yes')
else:
    print('No')