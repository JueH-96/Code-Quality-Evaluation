N,K=map(int,input().split())
X=[int(num)-1 for num in input().split()]
A=[int(num)-1 for num in input().split()]
G=[[] for _ in range(N)]
for i,x in enumerate(X):
    G[x].append(i)
d=[-1]*N
stack=[(i,1) for i in range(N) if not G[i]]
while stack:
    p,cycle=stack.pop()
    if d[p]==-1:
        d[p]=cycle
    elif d[p]==1:
        d[p]=cycle
        for np in G[p]:
            stack.append((np,cycle+1))
    else:
        l=cycle-d[p]
        d[p]=l
        for np in G[p]:
            stack.append((np,d[p]+1))
cycle=[[] for _ in range(N)]
for i in range(N):
    if not cycle[i]:
        q=[i]
        v=-1
        while q:
            v+=1
            p=q.pop(0)
            if d[p]>0:
                v%=d[p]
            cycle[p]=v
            for np in G[p]:
                if not cycle[np]:
                    q.append(np)
        v=2
res=[0]*N
for i in range(N):
    c=cycle[i]
    l=d[i]
    if l:
        if l>K:
            K=K%l
            p=i
            q=i
            for _ in range(K):
                p=X[p]
            res[i]=A[p]+1
        else:
            T=[(A[i]+1,1)]
            for _ in range(l):
                i=X[i]
                T.append((A[i]+1,T[-1][1]+1))
            q=0
            m=0
            while K:
                if K&1:
                    res[i]=T[q][0]
                    m+=T[q][1]
                q+=1
                if l-K>=0:
                    K=l-K
                else:
                    m%=l
                    K=K%l
                    p=i
                    for _ in range(m):
                        p=X[p]
                    i=p
                    T=[(A[i]+1,1)]
                    for _ in range(K):
                        i=X[i]
                        T.append((A[i]+1,T[-1][1]+1))
                    q=0
                K>>=1
    else:
        p=i
        for _ in range(K):
            p=X[p]
        res[i]=A[p]+1
print(" ".join(map(str,res)))