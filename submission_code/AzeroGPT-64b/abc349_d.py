import sys
L,R=map(int,sys.stdin.readline().split())
As=[]
while L<R:
    i=0;y=L
    while (1<<(i+1))<=R-L:y^=(1<<i);i+=1
    As.append((L,L|(1<<i)-1))
    L^=(1<<i)-1
print(len(As),*As,sep='
')