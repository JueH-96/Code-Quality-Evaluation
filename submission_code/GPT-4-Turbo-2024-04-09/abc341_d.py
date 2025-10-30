import sys
input = sys.stdin.read

def main():
    data = input().strip()
    N, M, K = map(int, data.split())
    
    # Function to count numbers divisible by N or M but not both up to x
    def count(x):
        divisible_by_N = x // N
        divisible_by_M = x // M
        divisible_by_both = x // (N * M // gcd(N, M))
        return divisible_by_N + divisible_by_M - 2 * divisible_by_both
    
    # Greatest common divisor
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Binary search for the K-th smallest number
    left, right = 1, 2
    # Increase right bound until it's surely above the K-th number
    while count(right) < K:
        right *= 2
    
    # Perform binary search between left and right
    while left < right:
        mid = (left + right) // 2
        if count(mid) < K:
            left = mid + 1
        else:
            right = mid
    
    print(left)

if __name__ == "__main__":
    main()