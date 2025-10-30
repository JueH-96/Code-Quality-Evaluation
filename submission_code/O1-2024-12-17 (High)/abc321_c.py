def main():
    import sys
    from itertools import combinations

    K = int(sys.stdin.readline().strip())

    digits = '0123456789'
    candidates = []

    # Generate all subsets of digits (1 to 10 in length)
    for length in range(1, 11):
        for combo in combinations(digits, length):
            # Exclude the single '0' subset (which is invalid as a positive integer)
            if length == 1 and combo[0] == '0':
                continue
            # Sort in descending order to form the 321-like number
            num_str = ''.join(sorted(combo, reverse=True))
            candidates.append(int(num_str))

    # Sort the candidates in ascending order
    candidates.sort()

    # Output the K-th smallest
    print(candidates[K - 1])

# Do not forget to call main()
if __name__ == "__main__":
    main()