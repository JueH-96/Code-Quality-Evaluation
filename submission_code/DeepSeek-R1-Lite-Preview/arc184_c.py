def count_trailing_ones(n):
    if n == 0:
        return 0
    count = 0
    while n & 1:
        count += 1
        n >>= 1
    return count

def is_mountain(n):
    return count_trailing_ones(n) % 2 == 1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    C = [a - 1 for a in A]
    
    from collections import defaultdict
    dp = defaultdict(int)
    dp[0] = 1
    for c in C:
        ndp = defaultdict(int)
        for mask, cnt in dp.items():
            ndp[mask ^ (1 << c)] += cnt
            ndp[mask] += cnt
        dp = ndp
    
    max_f = 0
    for mask, cnt in dp.items():
        count = 0
        tmp = mask
        while tmp:
            tmp &= tmp - 1
            count += 1
        if count % 2 == 1:
            max_f = max(max_f, count)
    print(max_f)

if __name__ == "__main__":
    main()