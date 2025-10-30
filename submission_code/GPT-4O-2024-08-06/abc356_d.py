# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    MOD = 998244353

    # Read inputs
    data = input().strip()
    N, M = map(int, data.split())

    # Initialize sum
    popcount_sum = 0

    # Iterate over all numbers from 0 to N
    for k in range(N + 1):
        # Compute k & M
        and_result = k & M
        # Compute popcount of the result
        popcount = bin(and_result).count('1')
        # Add to sum and take modulo
        popcount_sum = (popcount_sum + popcount) % MOD

    # Print the result
    print(popcount_sum)

if __name__ == "__main__":
    main()