# YOUR CODE HERE
import sys, math
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    LOG = 0
    tmp = K
    while tmp > 0:
        LOG += 1
        tmp >>=1
    up = [ [0]*N for _ in range(LOG)]
    for i in range(N):
        up[0][i] = X[i]-1
    for j in range(1, LOG):
        for i in range(N):
            up[j][i] = up[j-1][up[j-1][i]]
    res = [0]*N
    for i in range(N):
        pos = i
        k = K
        for j in range(LOG):
            if k & 1:
                pos = up[j][pos]
            k >>=1
            if k ==0:
                break
        res[i] = A[pos]
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()