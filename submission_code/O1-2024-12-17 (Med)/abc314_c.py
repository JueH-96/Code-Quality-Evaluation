def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = list(data[2])
    C = list(map(int, data[3:]))

    # Build a list of positions for each color
    color_positions = [[] for _ in range(M+1)]
    for i in range(N):
        color_positions[C[i]].append(i)

    # Perform the right circular shift for each color in order
    for color in range(1, M + 1):
        positions = color_positions[color]
        if len(positions) > 1:  # Only shift if there's more than one character of this color
            last_char = S[positions[-1]]
            for j in range(len(positions) - 1, 0, -1):
                S[positions[j]] = S[positions[j - 1]]
            S[positions[0]] = last_char

    print("".join(S))

# Call main() to execute the solution
if __name__ == "__main__":
    main()