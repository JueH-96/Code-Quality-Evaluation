import sys, bisect
from math import gcd

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N,M,C,K,*rest = map(int, sys.stdin.read().split())
    A = sorted(rest)
    g = gcd(C,M)
    T = M // g
    sum_period =0
    for j in range(N):
        low = max(M - A[j],0)
        if j ==0:
            high = M -1
        else:
            high = M - A[j-1] -1
        # Find multiples of g in [low, high]
        low_q = ((low + g -1)//g)*g
        high_q = (high//g)*g
        if low_q > high_q:
            continue
        count = (high_q - low_q)//g +1
        sum_j = A[j] * count
        # Sum of r from low_q to high_q with step g
        sum_r = (low_q + high_q) * count //2
        sum_j += sum_r
        # Subtract M * count
        sum_j -= M * count
        sum_period +=sum_j
    # Sum_rest: r < M - A[-1]
    upper = M - A[-1] -1
    if upper >=0:
        count_rest = (upper)//g +1
        sum_rest = count_rest * A[0] + g * count_rest * (count_rest -1) //2
    else:
        sum_rest =0
    sum_period +=sum_rest
    num_periods = K // T
    remaining = K % T
    total = num_periods * sum_period
    if remaining >0:
        for k in range(remaining):
            r = (C *k) % M
            if r >= M - A[-1]:
                pos = bisect.bisect_left(A, M - r)
                if pos <N:
                    val = A[pos] + r - M
                else:
                    val = A[0] + r
            else:
                val = A[0] + r
            total +=val
    print(total)

if __name__ == "__main__":
    main()