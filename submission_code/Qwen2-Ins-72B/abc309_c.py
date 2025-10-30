import sys
import math
from collections import defaultdict,Counter

# input=sys.stdin.readline
# def print(x):
#     sys.stdout.write(str(x)+"
")

# sys.stdout=open("CP1/output.txt",'w')
# sys.stdin=open("CP1/input.txt",'r')

# mod=pow(10,9)+7
n,k=list(map(int,input().split()))
l=[]
for i in range(n):
	a,b=list(map(int,input().split()))
	l.append([a,b])
l.sort()
i=0
ans=0
while i<n and l[i][1]>k:
	ans+=l[i][0]
	i+=1
if i==n:
	print(ans)
	return
if i<n:
	ans+=l[i][0]
	k-=l[i][1]
i+=1
while i<n:
	ans+=math.ceil(k/l[i][1])*l[i][0]
	print(ans)
	break
	i+=1