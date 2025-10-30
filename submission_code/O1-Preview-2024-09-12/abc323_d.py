# YOUR CODE HERE

import sys
import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    counts = {}
    for _ in range(N):
        S_i, C_i = map(int, sys.stdin.readline().split())
        counts[S_i] = counts.get(S_i, 0) + C_i

    sizes = sorted(counts.keys())

    max_iterations = 60  # Since counts up to 1e9, log2(1e9) ~ 30
    for _ in range(max_iterations):
        sizes_next = set()
        changed = False  # Flag to check if any counts change in this iteration
        new_sizes = []
        for S in sizes:
            counts_S = counts[S]
            P = counts_S // 2
            counts_S %= 2
            counts[S] = counts_S
            if P > 0:
                counts[2*S] = counts.get(2*S, 0) + P
                if 2*S not in counts or counts[2*S] - P == 0:
                    new_sizes.append(2*S)
                changed = True
        if not changed:
            break
        sizes.extend(new_sizes)
        sizes = sorted(set(sizes))
    total_slimes = sum(counts[S] for S in counts)
    print(total_slimes)

threading.Thread(target=main,).start()