# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.read().strip()
    from collections import defaultdict

    char_positions = defaultdict(list)
    for idx, c in enumerate(S, 1):
        char_positions[c].append(idx)

    total = 0
    for c in char_positions:
        pos = char_positions[c]
        m = len(pos)
        if m <2:
            continue
        prefix_sum = [0]*m
        prefix_sum[0] = pos[0]
        for i in range(1,m):
            prefix_sum[i] = prefix_sum[i-1] + pos[i]
        for j in range(1,m):
            # j is current index (0-based)
            # number of previous positions: j
            # sum of previous positions: prefix_sum[j-1]
            contribution = j * pos[j] - prefix_sum[j-1] - j
            total += contribution
    print(total)

if __name__ == "__main__":
    main()