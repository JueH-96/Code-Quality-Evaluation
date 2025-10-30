def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Calculate the partial sums and track the minimum partial sum
    partial_sum = 0
    min_partial_sum = float('inf')
    for a in A:
        partial_sum += a
        if partial_sum < min_partial_sum:
            min_partial_sum = partial_sum

    # If the minimum partial sum is already >= 0, the bus can start with 0 passengers
    # Otherwise, we need an initial count so that p[i] = initial + partial_sum[i] >= 0
    # => initial >= -min_partial_sum
    # The answer is final_partial_sum + max(0, -min_partial_sum).
    final_sum = partial_sum
    offset = 0 if min_partial_sum >= 0 else -min_partial_sum
    print(final_sum + offset)

# Do not remove the call to main()
main()