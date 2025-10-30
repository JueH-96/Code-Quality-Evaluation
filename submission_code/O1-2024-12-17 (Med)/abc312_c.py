def main():
    import sys
    data = list(map(int, sys.stdin.read().strip().split()))
    N, M = data[0], data[1]
    A = data[2:2+N]
    B = data[2+N:2+N+M]

    A.sort()
    B.sort()

    from bisect import bisect_left, bisect_right

    # S(X) = number of sellers who can sell at price X = count of A_i <= X
    def S(X):
        return bisect_right(A, X)
    
    # Bcount(X) = number of buyers who can buy at price X = count of B_i >= X
    def Bcount(X):
        return M - bisect_left(B, X)
    
    # We'll binary search for the minimal X in [1 .. 10^9+1]
    low, high = 1, 10**9 + 1
    while low < high:
        mid = (low + high) // 2
        if S(mid) >= Bcount(mid):
            high = mid
        else:
            low = mid + 1
    
    print(low)

# Do not forget to call main()!
main()