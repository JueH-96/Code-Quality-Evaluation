# YOUR CODE HERE
def solve():
    import sys
    import sys
    def input():
        return sys.stdin.read()
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr +=1
    A = list(map(int, data[ptr:ptr+N]))
    ptr +=N
    S = []
    T = []
    for _ in range(N-1):
        s = int(data[ptr])
        t = int(data[ptr+1])
        S.append(s)
        T.append(t)
        ptr +=2
    for i in range(N-1):
        k = A[i] // S[i]
        if k >0:
            A[i] -= k * S[i]
            A[i+1] += k * T[i]
    print(A[-1])