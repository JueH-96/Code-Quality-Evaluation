import math

def main():
    N, M, K = map(int, input().split())
    g = math.gcd(N, M)
    L = (N // g) * M
    
    def count_valid(x):
        return x // N + x // M - 2 * (x // L)
    
    low, high = 1, 10**20
    while low < high:
        mid = (low + high) // 2
        if count_valid(mid) >= K:
            high = mid
        else:
            low = mid + 1
    print(low)

if __name__ == '__main__':
    main()