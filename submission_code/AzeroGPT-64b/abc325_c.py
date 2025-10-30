from collections import deque
H,W=map(int,input().split())
s=[input() for _ in range(H)]
num=0
visit=[[False]*W for _ in range(H)]
for j in range(H):
    for k in range(W):
        if visit[j][k]==False and s[j][k]=="#":
            num+=1
            q=deque([[j,k]])
            while q:
                a,b=q.popleft()
                visit[a][b]=True
                for x in [-1,0,1]:
                    for y in [-1,0,1]:
                        if a+x<H and a+x>=0 and b+y<W and b+y>=0:
                            if visit[a+x][b+y]==False and s[a+x][b+y]=="#":
                                visit[a+x][b+y]=True
                                q.append([a+x,b+y])
print(num)