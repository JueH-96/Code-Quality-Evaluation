import math

def main():
    N, M, K = map(int, input().split())
    
    # Compute GCD and LCM
    gcd = math.gcd(N, M)
    lcm = N * M // gcd
    
    # Binary search
    low = 1
    high = K * max(N, M)
    
    while low < high:
        mid = (low + high) // 2
        count = (mid // N) + (mid // M) - 2 * (mid // lcm)
        if count < K:
            low = mid + 1
        else:
            high = mid
    
    # Verify that low is the K-th number
    count = (low // N) + (low // M) - 2 * (low // lcm)
    if count >= K:
        print(low)
    else:
        # If low is not sufficient, increment to find the correct K-th number
        low += 1
        while (low // N) + (low // M) - 2 * (low // lcm) < K:
            low += 1
        print(low)

if __name__ == "__main__":
    main()