import sys

def read_ints(): return list(map(int, sys.stdin.readline().strip().split()))

def solve():
    N = int(input())
    P = [read_ints() for _ in range(N)]
    Q = [read_ints() for _ in range(N)]
    
    # Sort P and Q based on x-coordinate
    P.sort()
    Q.sort()
    
    # Create a mapping from P to Q based on the sorted order
    R = [Q.index(q) + 1 for p, q in zip(P, Q)]
    
    print(' '.join(map(str, R)))

if __name__ == "__main__":
    solve()