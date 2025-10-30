def solve():
    import sys
    input_data = sys.stdin.read().strip()
    K = int(input_data)

    digits = list(range(10))  # 0 through 9

    # We will collect all 321-like numbers here
    all_numbers = []

    # Generate all subsets of {0..9} of non-empty size
    # Then build numbers from those subsets if they include at least one of {1..9} as the highest digit
    from itertools import combinations
    for length in range(1, 11):  # subset sizes from 1 to 10
        for combo in combinations(digits, length):
            # combo is a subset of digits; sort them descending to form a 321-like number
            sorted_combo = sorted(combo, reverse=True)
            # If the first (largest) digit is 0, it wouldn't form a positive integer (leading zero)
            if sorted_combo[0] == 0:
                continue
            # Construct the number
            val = 0
            for d in sorted_combo:
                val = val * 10 + d
            all_numbers.append(val)

    all_numbers.sort()
    print(all_numbers[K - 1])

def main():
    solve()

if __name__ == "__main__":
    main()