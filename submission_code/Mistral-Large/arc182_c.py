import sys

def count_divisors(n):
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += (1 if i == n // i else 2)
    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    MOD = 998244353

    # Precompute the number of divisors for all products up to M^N
    max_product = M ** N
    divisors = [0] * (max_product + 1)

    for i in range(1, max_product + 1):
        divisors[i] = count_divisors(i)

    # Initialize the total score
    total_score = 0

    # Iterate over all possible sequence lengths
    for length in range(1, N + 1):
        # Iterate over all possible sequences of the given length
        for seq in range(1, M**length + 1):
            product = 1
            temp = seq
            for _ in range(length):
                product *= (temp % M)
                temp //= M
            total_score += divisors[product]
            total_score %= MOD

    print(total_score)

if __name__ == "__main__":
    main()