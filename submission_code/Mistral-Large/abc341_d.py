import sys
from math import gcd

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    K = int(data[2])

    lcm_nm = N * M // gcd(N, M)

    low, high = 0, 2 * max(N, M) * K

    while high - low > 1:
        mid = (low + high) // 2
        count = (mid // N) + (mid // M) - (mid // lcm_nm)
        if count >= K:
            high = mid
        else:
            low = mid

    print(high)

if __name__ == "__main__":
    main()