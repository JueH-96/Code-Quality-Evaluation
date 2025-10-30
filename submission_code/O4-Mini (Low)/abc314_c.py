import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:]))

    # Collect positions for each color
    positions = [[] for _ in range(M + 1)]
    for idx, color in enumerate(C):
        positions[color].append(idx)

    # For each color, perform a right circular shift on its positions
    for color in range(1, M + 1):
        pos_list = positions[color]
        k = len(pos_list)
        if k <= 1:
            continue
        # Save last character
        last_char = S[pos_list[-1]]
        # Shift right by one
        for j in range(k - 1, 0, -1):
            S[pos_list[j]] = S[pos_list[j - 1]]
        S[pos_list[0]] = last_char

    # Output result
    sys.stdout.write("".join(S))

if __name__ == "__main__":
    main()