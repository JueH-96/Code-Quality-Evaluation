def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Dictionary to store positions of each distinct value
    from collections import defaultdict
    pos_map = defaultdict(list)

    # Collect positions (1-based indexing) for each value
    for i, val in enumerate(A, start=1):
        pos_map[val].append(i)
    
    # We'll add a 0 at the start and N+1 at the end
    # Then use the formula Σ ( (p_m - p_{m-1}) * (p_{m+1} - p_m ) )
    # over m = 1..k for each distinct value's positions
    ans = 0
    for val, positions in pos_map.items():
        # Insert boundary positions
        positions = [0] + positions + [N+1]
        for i in range(1, len(positions) - 1):
            ans += (positions[i] - positions[i-1]) * (positions[i+1] - positions[i])

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()