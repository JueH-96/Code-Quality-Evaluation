MOD = 998244353

def main():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    # Prefix sums and their counts
    prefix_sums = {0: 1}
    current_sum = 0
    valid_divisions = 1

    for i in range(N):
        current_sum += A[i]
        # Check if there is a subsequence ending at the current position
        # that sums up to K by looking at prefix_sums
        if current_sum - K in prefix_sums:
            valid_divisions -= prefix_sums[current_sum - K]
            valid_divisions %= MOD

        # Update the count of prefix sums
        if current_sum in prefix_sums:
            prefix_sums[current_sum] += 1
        else:
            prefix_sums[current_sum] = 1

        # Update the total count of valid divisions
        valid_divisions *= 2
        valid_divisions %= MOD

    print(valid_divisions)

if __name__ == "__main__":
    main()