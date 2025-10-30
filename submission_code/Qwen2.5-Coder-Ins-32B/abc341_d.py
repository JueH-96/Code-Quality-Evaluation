# YOUR CODE HERE
import sys

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisibles(n, m, limit):
    lcm_nm = lcm(n, m)
    return (limit // n) + (limit // m) - (limit // lcm_nm)

def find_kth_smallest(n, m, k):
    low, high = 1, k * min(n, m)
    while low < high:
        mid = (low + high) // 2
        if count_divisibles(n, m, mid) < k:
            low = mid + 1
        else:
            high = mid
    return low

def main():
    import sys
    input = sys.stdin.read().strip().split()
    N = int(input[0])
    M = int(input[1])
    K = int(input[2])
    result = find_kth_smallest(N, M, K)
    print(result)

if __name__ == "__main__":
    main()