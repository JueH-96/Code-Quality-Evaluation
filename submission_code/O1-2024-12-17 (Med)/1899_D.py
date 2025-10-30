def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    out = []

    for _ in range(t):
        n = int(input_data[idx]); idx += 1
        arr = list(map(int, input_data[idx: idx + n]))
        idx += n

        # Count frequencies of each a_i
        freq = {}
        for val in arr:
            freq[val] = freq.get(val, 0) + 1

        # Count pairs where a_i = a_j
        total_pairs = 0
        for c in freq.values():
            if c > 1:
                total_pairs += c * (c - 1) // 2

        # Add pairs (1,2) if they exist
        c1 = freq.get(1, 0)
        c2 = freq.get(2, 0)
        total_pairs += c1 * c2

        out.append(str(total_pairs))

    print("
".join(out))

# Do not forget to call main()
main()