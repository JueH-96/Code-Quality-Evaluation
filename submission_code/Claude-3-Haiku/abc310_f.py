import math

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Compute the probability modulo 998244353
    total_ways = 1
    for a in A:
        total_ways = (total_ways * a) % 998244353

    # Compute the number of ways to choose some dice such that the sum is 10
    ways_to_sum_10 = 0
    for mask in range(1, (1 << N)):
        curr_sum = 0
        for i in range(N):
            if mask & (1 << i):
                curr_sum += i + 1
        if curr_sum == 10:
            ways_to_sum_10 += 1

    # Compute the answer modulo 998244353
    answer = (ways_to_sum_10 * pow(total_ways, 998244351, 998244353)) % 998244353
    print(answer)

solve()