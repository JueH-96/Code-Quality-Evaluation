def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    strings = data[2:]

    # Convert each stand's popcorn flavors into a bitmask
    stands = []
    for s in strings:
        mask = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                mask |= 1 << j
        stands.append(mask)

    full_mask = (1 << M) - 1  # bitmask representing all flavors
    answer = N  # worst case: visit all stands

    # Enumerate all subsets of stands
    for subset in range(1, 1 << N):
        combined_mask = 0
        for i in range(N):
            if (subset >> i) & 1:
                combined_mask |= stands[i]
        # Check if this subset covers all flavors
        if combined_mask == full_mask:
            # Calculate the number of stands in this subset
            count = bin(subset).count('1')
            answer = min(answer, count)

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()