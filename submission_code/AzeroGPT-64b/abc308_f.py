from heapq import heapify,heappush,heappop

N,M=map(int,input().split())

P=list(map(int,input().split()))

L=list(map(int,input().split()))

D=list(map(int,input().split()))

P.sort()

li=[[-D[i],-L[i]] for i in range(M)]

heapify(li)

ans=sum(P)

heapify(P)

for i in range(M):
  heappush(P,heappop(P)-heappop(li)[0])

print(ans+sum(P))