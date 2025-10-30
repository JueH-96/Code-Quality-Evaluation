def main():
    import sys
    import bisect
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    A.sort()
    B.sort()
    
    # f(x) = (number of sellers who can sell at price x) - (number of buyers who can buy at price x)
    #       = (# of A_i <= x) - (# of B_i >= x)
    def f(x):
        # number of sellers who require <= x
        sellers = bisect.bisect_right(A, x)
        # number of buyers who can pay >= x
        buyers = M - bisect.bisect_left(B, x)
        return sellers - buyers
    
    left, right = 1, 10**9 + 2  # 1 <= A_i, B_i <= 10^9, so go a bit beyond
    while left < right:
        mid = (left + right) // 2
        if f(mid) >= 0:
            right = mid
        else:
            left = mid + 1
    
    print(left)

# call main
if __name__ == "__main__":
    main()