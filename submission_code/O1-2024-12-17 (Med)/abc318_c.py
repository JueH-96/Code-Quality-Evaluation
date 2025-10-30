def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    fares = list(map(int, data[3:]))

    # Sort fares descending
    fares.sort(reverse=True)

    # Compute prefix sums of fares in descending order
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + fares[i]

    total_fare = prefix_sum[N]
    answer = total_fare  # Worst case: pay all fares without passes

    # For each possible number of days covered by passes (m = 0..N)
    for m in range(N + 1):
        # Sum of top m fares
        covered_sum = prefix_sum[m]
        # Pass cost: number of pass batches * P
        pass_cost = ((m + D - 1) // D) * P
        # Regular fare for the rest of days not covered by passes
        regular_cost = total_fare - covered_sum
        # Total cost
        cost = pass_cost + regular_cost
        if cost < answer:
            answer = cost

    print(answer)

# Call main function
if __name__ == "__main__":
    main()