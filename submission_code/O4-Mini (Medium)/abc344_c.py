import sys
import threading

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    # Read sizes and arrays
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = int(next(it))
    B = [int(next(it)) for _ in range(M)]
    L = int(next(it))
    C = [int(next(it)) for _ in range(L)]
    Q = int(next(it))
    X = [int(next(it)) for _ in range(Q)]
    
    # Precompute all sums of A and B
    # Size at most 100*100 = 10,000
    S = set()
    for a in A:
        for b in B:
            S.add(a + b)
    
    # For each query X_i, check if there exists c in C such that X_i - c in S
    out = []
    S_contains = S.__contains__  # local reference for speed
    for x in X:
        ok = False
        # loop over C (<= 100)
        for c in C:
            if S_contains(x - c):
                ok = True
                break
        out.append("Yes" if ok else "No")
    
    # Print results
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()