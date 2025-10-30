from math import sqrt

def num_divisors(n):
    count = 0
    for i in range(1, int(sqrt(n)) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def solve(N, M):
    total_score = 0
    for k in range(1, N+1):
        total_score += M**k * num_divisors(M**k)
    return total_score % 998244353

if __name__ == "__main__":
    N, M = map(int, input().split())
    print(solve(N, M))