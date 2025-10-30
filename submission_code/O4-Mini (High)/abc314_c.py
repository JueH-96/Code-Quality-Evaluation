import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N, M = map(int, data)
    S = list(sys.stdin.readline().rstrip())
    C = list(map(int, sys.stdin.readline().split()))

    # Collect positions for each color (1-indexed)
    positions = [[] for _ in range(M + 1)]
    for idx, color in enumerate(C):
        positions[color].append(idx)

    # For each color, do a right circular shift of the chars at those positions
    for color in range(1, M + 1):
        pos_list = positions[color]
        k = len(pos_list)
        if k <= 1:
            continue
        # Save the last character
        last_char = S[pos_list[-1]]
        # Shift others to the right by one
        for j in range(k - 1, 0, -1):
            S[pos_list[j]] = S[pos_list[j - 1]]
        # Place the saved character at the first position
        S[pos_list[0]] = last_char

    # Output the final string
    sys.stdout.write(''.join(S))

if __name__ == "__main__":
    main()