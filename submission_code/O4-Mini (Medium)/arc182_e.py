#!/usr/bin/env python3
import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    C = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    A.sort()
    # floor_sum: sum_{i=0..n-1} floor((a*i + b) / m)
    def floor_sum(n, m, a, b):
        # from AtCoder Library
        res = 0
        # allow b >= m or a >= m
        if a >= m:
            # sum_{i=0..n-1} floor(a/m * i) = (a//m) * n*(n-1)//2
            q = a // m
            res += q * (n*(n-1)//2)
            a %= m
        if b >= m:
            q = b // m
            res += q * n
            b %= m
        # now a < m, b < m
        # main loop
        while n:
            # max value of a*(n-1) + b
            y = a*n + b
            if y < m:
                break
            # we will change dims
            # let n' = y // m
            n_new = y // m
            b = y % m
            # swap a,m
            # sum floor((a*i + b)/m) = sum_{j=0..n'-1} floor((m*j + b')/a) with roles swapped
            # add n' * (n' - 1) // 2 * ((m)//a)
            # Actually we use recursion via variables exchange
            # but direct formula: res += (n_new) * (b // a)
            # Use the standard transformation:
            # res += (n_new) * (b // a) # WRONG
            # Actually the AtCoder loop:
            # tmp = m // a
            # m, a = a, m % a
            # b = b
            # res += tmp * n_new*(n_new-1)//2
            # But that's the same pattern as first a>=m.
            # Use the typical code:
            # Swap a,m and set n=b_new...
            # Let's just use the known code:
            # At this point, y//m = n_new, and we will set:
            # res += (n_new)*(b)//a ??
            # Actually refer to ACL:
            #   n = y//m
            #   b = y%m
            #   swap(m,a)
            #   continue
            # And before loop, handle a>=m,b>=m. So we do just that.
            # But we also need to add something? The ACL code adds nothing here.
            # So implement direct:
            # Swap m,a, set n,b.
            # But that loses floorsum contributions?
            # Actually ACL code:
            #   while 1:
            #       if a >= m:
            #           res += (n-1)*n*(a//m)//2
            #           a %= m
            #       if b >= m:
            #           res += n*(b//m)
            #           b %= m
            #       y_max = a*n + b
            #       if y_max < m: break
            #       n = y_max//m
            #       b = y_max%m
            #       a, m = m, a
            #   return res
            # So we follow exactly:
            n = n_new
            m, a = a, m
        return res

    # compute S1 = sum floor(C*i / M)
    S1 = floor_sum(K, M, C, 0)
    # sum of t_k = sum (C*k mod M) = C*K*(K-1)//2 - M * S1
    sum_t = C * (K*(K-1)//2) - M * S1
    # prepare bs array: bs[i] = floor_sum(K,M,C, A_ext[i])
    # A_ext[0] = 0, A_ext[1..N] = A[0..N-1]
    # bs[0] = floor_sum(K,M,C,0) = S1
    bs = [0]*(N+1)
    bs[0] = S1
    # compute bs[1..N]
    for i in range(1, N+1):
        b = A[i-1]
        bs[i] = floor_sum(K, M, C, b)
    # bs_M = floor_sum(K,M,C,M) = S1 + K
    bs_M = S1 + K
    ans = sum_t
    # interval 0: cnt0 = bs_M - bs[N], d0 = A[0]
    cnt0 = bs_M - bs[N]
    if N>0:
        d0 = A[0]
        ans += cnt0 * d0
    # intervals 1..N: for i in 1..N: cnt = bs[i] - bs[i-1], d = A[i-1] - M
    for i in range(1, N+1):
        cnt = bs[i] - bs[i-1]
        d = A[i-1] - M
        ans += cnt * d
    # print answer
    print(ans)

if __name__ == "__main__":
    main()