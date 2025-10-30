MOD = 998244353

def count_divisors(n):
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def main():
    import sys
    input = sys.stdin.read
    N, M = map(int, input().split())

    total_score = 0
    for length in range(1, N + 1):
        for num in range(1, M + 1):
            total_score += count_divisors(num)
            total_score %= MOD

    print(total_score)

if __name__ == "__main__":
    main()