# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))

    pos = {}
    for idx, val in enumerate(A):
        if val in pos:
            pos[val].append(idx)
        else:
            pos[val] = [idx]
    ans = 0
    for val in pos:
        positions = pos[val]
        freq = len(positions)
        if freq < 2:
            continue
        positions.sort()
        cumsum = [0]
        for p in positions:
            cumsum.append(cumsum[-1]+p)

        total_triples = 0
        for i in range(freq -1):
            sum_after_i = cumsum[-1] - cumsum[i+1]
            count = freq - i -1
            total_triples += sum_after_i - positions[i]*count - count

        total_overlaps = freq*(freq-1)*(freq-2)//6 if freq>=3 else 0
        ans += total_triples - total_overlaps
    print(ans)

threading.Thread(target=main).start()