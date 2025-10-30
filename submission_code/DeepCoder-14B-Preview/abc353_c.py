import bisect

def main():
    MOD = 10**8
    n = int(input())
    a = list(map(int, input().split()))
    
    # Compute B_i = A_i mod 1e8
    b = [x % MOD for x in a]
    b.sort()
    
    sum_b = sum(b)
    count = 0
    
    for i in range(n - 1):
        target = MOD - b[i]
        # Find the first index j > i where b[j] >= target
        j = bisect.bisect_left(b, target, i + 1, n)
        count += (n - j)
    
    total = sum_b * (n - 1) - count * MOD
    print(total)

if __name__ == '__main__':
    main()