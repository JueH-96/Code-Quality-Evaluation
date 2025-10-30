import math

def main():
    import sys
    N, M, K = map(int, sys.stdin.read().split())
    
    # Compute GCD and LCM
    gcd_nm = math.gcd(N, M)
    lcm_nm = (N * M) // gcd_nm
    
    # Function to count numbers divisible by exactly one of N or M up to X
    def count(X):
        return X // N + X // M - 2 * (X // lcm_nm)
    
    # Binary search to find the smallest X where count(X) >= K
    left = 1
    right = K * max(N, M)
    
    while left < right:
        mid = (left + right) // 2
        cnt = count(mid)
        if cnt >= K:
            right = mid
        else:
            left = mid + 1
    print(left)

if __name__ == "__main__":
    main()