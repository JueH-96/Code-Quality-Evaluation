def main():
    import sys
    from itertools import accumulate

    S = sys.stdin.read().strip()
    n = len(S)
    if n < 3:
        print(0)
        return

    # Initialize lists for each character 'A' to 'Z'
    pos = [[] for _ in range(26)]
    for idx, c in enumerate(S, 1):  # positions are 1-based
        pos[ord(c) - ord('A')].append(idx)

    total = 0
    for char_pos in pos:
        m = len(char_pos)
        if m < 2:
            continue
        # Compute prefix sums
        prefix_sum = list(accumulate(char_pos, initial=0))
        sum_pairs = 0
        for b in range(1, m):
            sum_pairs += char_pos[b] * b - prefix_sum[b]
        # Subtract m choose 2
        total += sum_pairs - (m * (m - 1) // 2)
    
    print(total)

if __name__ == "__main__":
    main()