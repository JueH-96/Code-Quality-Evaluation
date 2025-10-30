def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, D, P = map(int, data[:3])
    fares = list(map(int, data[3:]))

    # Sort fares in descending order
    fares.sort(reverse=True)

    # Compute prefix sums
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i + 1] = prefix_sums[i] + fares[i]

    # This is the total of all fares (paying regular fare for every day)
    total_fares = prefix_sums[N]

    # We'll try buying x batches of D passes for x = 0 .. (N + D - 1) // D
    max_batches = (N + D - 1) // D
    answer = total_fares  # case of using no day passes

    for x in range(1, max_batches + 1):
        # If x*D exceeds N, we only need to cover N days at most
        days_covered = x * D
        if days_covered > N:
            days_covered = N

        # Cost if we buy x batches of passes:
        #   = x*P (the cost of the passes)
        #     + (total_fares - prefix_sums[days_covered]) (the cost of days not covered by passes)
        cost = x * P + (total_fares - prefix_sums[days_covered])
        answer = min(answer, cost)

    print(answer)

# Don't forget to call main function
if __name__ == "__main__":
    main()