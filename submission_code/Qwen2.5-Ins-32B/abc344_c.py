import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    m = int(data[n+1])
    B = list(map(int, data[n+2:n+m+2]))
    l = int(data[n+m+2])
    C = list(map(int, data[n+m+3:n+m+l+3]))
    q = int(data[n+m+l+3])
    X = list(map(int, data[n+m+l+4:n+m+l+q+4]))
    
    AB = set(a + b for a in A for b in B)
    
    for x in X:
        if any(x - c in AB for c in C):
            print('Yes')
        else:
            print('No')

solve()