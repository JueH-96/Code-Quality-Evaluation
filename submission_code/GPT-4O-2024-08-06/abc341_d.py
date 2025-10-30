import sys
import math

def count_exactly_one(x, N, M, lcm_nm):
    count_N = x // N
    count_M = x // M
    count_both = x // lcm_nm
    return count_N + count_M - 2 * count_both

def find_kth_smallest(N, M, K):
    lcm_nm = (N * M) // math.gcd(N, M)
    
    low, high = 1, 10**18  # Start with a high value to ensure we cover large K
    while low < high:
        mid = (low + high) // 2
        if count_exactly_one(mid, N, M, lcm_nm) < K:
            low = mid + 1
        else:
            high = mid
    return low

def main():
    input = sys.stdin.read
    data = input().strip()
    N, M, K = map(int, data.split())
    result = find_kth_smallest(N, M, K)
    print(result)

if __name__ == "__main__":
    main()