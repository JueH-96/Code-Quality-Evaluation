import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    # Parse inputs
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:]))

    # Build list of positions for each color 1..M
    positions = [[] for _ in range(M+1)]
    for idx, color in enumerate(C):
        positions[color].append(idx)

    # For each color in order, perform right circular shift by 1
    for color in range(1, M+1):
        pos_list = positions[color]
        k = len(pos_list)
        if k <= 1:
            continue  # no change if there's only one (or zero) position
        
        # Extract the characters in these positions
        seq = [S[i] for i in pos_list]
        # Rotate right by 1
        last = seq[-1]
        # Assign back in rotated order
        for j in range(k):
            # The new j-th value comes from seq[(j-1) % k],
            # but we already stored last, so handle j=0 specially
            if j == 0:
                S[pos_list[0]] = last
            else:
                S[pos_list[j]] = seq[j-1]

    # Print the final string
    sys.stdout.write("".join(S))


if __name__ == "__main__":
    main()