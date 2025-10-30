# YOUR CODE HERE
def solve():
    N = int(input())
    X = list(map(int, input().split()))

    # Step 1: Calculate initial differences
    # d_k corresponds to diffs[k-1] in 0-indexed list for k in [1, N-1]
    # diffs[j] = X[j+1] - X[j] for j in [0, N-2]
    diffs = [X[k+1] - X[k] for k in range(N - 1)]

    # Step 2: Separate differences into odd and even indexed lists
    # odd_diffs contains d_1, d_3, d_5, ... (original 0-indexed diffs[0], diffs[2], ...)
    odd_diffs = diffs[::2]
    # even_diffs contains d_2, d_4, d_6, ... (original 0-indexed diffs[1], diffs[3], ...)
    even_diffs = diffs[1::2]

    # Step 3 & 4: Sort the separated lists in ascending order
    odd_diffs.sort()
    even_diffs.sort()

    # Step 5: Reconstruct the optimal d' sequence
    d_prime = [0] * (N - 1)
    odd_idx = 0
    even_idx = 0
    for k in range(N - 1):
        # The original d_k is at index k in 'diffs'.
        # We need to decide if current d_prime[k] should be from odd_diffs or even_diffs.
        # This depends on the parity of (k+1), which is the 1-based index of d.
        if (k + 1) % 2 != 0:  # If the original d_k (1-based index) was odd
            d_prime[k] = odd_diffs[odd_idx]
            odd_idx += 1
        else:  # If the original d_k (1-based index) was even
            d_prime[k] = even_diffs[even_idx]
            even_idx += 1

    # Step 6: Calculate the final sum of coordinates
    # The first coordinate X_1 (X[0]) remains fixed.
    current_X_value = X[0]
    total_sum = current_X_value

    # Iterate through d_prime to reconstruct subsequent X values and add to total sum
    for diff in d_prime:
        current_X_value += diff
        total_sum += current_X_value

    print(total_sum)

solve()