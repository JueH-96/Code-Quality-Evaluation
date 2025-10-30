def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    # Parse inputs
    N, M = map(int, data[:2])
    S = list(data[2])
    colors = list(map(int, data[3:]))

    # Build list of positions for each color
    positions = [[] for _ in range(M)]
    for i in range(N):
        positions[colors[i] - 1].append(i)

    # For each color, right-shift the characters in their respective positions
    for i in range(M):
        p = positions[i]
        if len(p) > 1:
            # Extract the characters
            old_vals = [S[idx] for idx in p]
            # Perform right circular shift by 1
            shifted = old_vals[-1:] + old_vals[:-1]
            # Place them back
            for j in range(len(p)):
                S[p[j]] = shifted[j]

    # Output the final result
    print("".join(S))

# Do not forget to call main()!
if __name__ == "__main__":
    main()