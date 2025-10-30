import math

def main():
    N, M, K = map(int, input().split())
    gcd = math.gcd(N, M)
    lcm = N * M // gcd
    low = 1
    high = K * max(N, M)
    
    def count(x):
        a = x // N
        b = x // M
        c = x // lcm
        return a + b - 2 * c
    
    while low < high:
        mid = (low + high) // 2
        cnt = count(mid)
        if cnt < K:
            low = mid + 1
        else:
            high = mid
    print(low)

if __name__ == "__main__":
    main()