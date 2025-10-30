def main():
    import sys
    input_data = sys.stdin.read().strip()
    N = int(input_data)

    # Generate a list of repunits up to 40 digits
    repunits = []
    current = 0
    for _ in range(40):
        current = current * 10 + 1
        repunits.append(current)

    # Generate all possible sums of exactly three repunits
    sums_set = set()
    for i in range(40):
        for j in range(40):
            for k in range(40):
                sums_set.add(repunits[i] + repunits[j] + repunits[k])

    # Sort the distinct sums
    sorted_sums = sorted(sums_set)

    # Output the N-th smallest (1-indexed)
    print(sorted_sums[N - 1])

if __name__ == "__main__":
    main()