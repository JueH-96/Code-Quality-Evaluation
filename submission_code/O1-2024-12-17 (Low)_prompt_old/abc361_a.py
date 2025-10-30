def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K, X = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    
    B = A[:K] + [X] + A[K:]
    print(" ".join(map(str, B)))

solve()