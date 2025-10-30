import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    
    # Convert X to 0-based permutation
    f = [x - 1 for x in X]
    
    def compose(g, h):
        res = [0] * N
        for i in range(N):
            res[i] = g[h[i]]
        return res
    
    # Initialize result as identity function
    result = list(range(N))
    current = f.copy()
    exponent = K
    
    while exponent > 0:
        if exponent % 2 == 1:
            result = compose(result, current)
        current = compose(current, current)
        exponent = exponent // 2
    
    ans = [A[result[i]] for i in range(N)]
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()