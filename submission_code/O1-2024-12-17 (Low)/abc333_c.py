def main():
    import sys
    input_data = sys.stdin.read().strip()
    N = int(input_data)

    # We will generate repunits up to a certain number of digits.
    # Based on the sample outputs (e.g., the 333rd sum is already 11 digits),
    # we'll go up to, say, 15-digit repunits for a safe margin.
    max_digits = 15

    # Generate repunits (all ones) up to max_digits.
    repunits = []
    current = 0
    for d in range(1, max_digits + 1):
        current = current * 10 + 1  # Build 1, 11, 111, ...
        repunits.append(current)

    # Generate all possible sums of exactly three repunits.
    possible_sums = set()
    length = len(repunits)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                s = repunits[i] + repunits[j] + repunits[k]
                possible_sums.add(s)

    # Sort the sums in ascending order and pick the N-th smallest
    sorted_sums = sorted(possible_sums)
    print(sorted_sums[N-1])

# Do not forget to call main()
if __name__ == "__main__":
    main()