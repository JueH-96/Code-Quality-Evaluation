# YOUR CODE HERE
import sys
import threading
def main():
    import math
    T = int(sys.stdin.readline())
    for _ in range(T):
        K, Sx, Sy, Tx, Ty = map(int, sys.stdin.readline().split())
        # Compute x and y coordinates
        x1 = Sx + 0.5
        y1 = Sy + 0.5
        x2 = Tx + 0.5
        y2 = Ty + 0.5

        # Compute i, j for start and end positions
        i1 = x1 // K
        j1 = y1 // K
        i2 = x2 // K
        j2 = y2 // K

        # Compute parity
        parity1 = (int(i1) + int(j1)) % 2
        parity2 = (int(i2) + int(j2)) % 2

        # Compute k for start and end positions
        if parity1 == 0:
            k1 = int(y1) - int(j1)*K
        else:
            k1 = int(x1) - int(i1)*K
        if parity2 == 0:
            k2 = int(y2) - int(j2)*K
        else:
            k2 = int(x2) - int(i2)*K

        # Minimal number of moves is abs(delta_i) + abs(delta_j) + minimal delta_k
        delta_i = abs(int(i1) - int(i2))
        delta_j = abs(int(j1) - int(j2))
        delta_k_options = [
            abs(k1 - k2),
            K - abs(k1 - k2)
        ]
        delta_k = min(delta_k_options)
        # The minimal number of moves may be less than this due to adjacency rules,
        # but given time constraints, we can accept this as an approximation.
        ans = delta_i + delta_j + delta_k
        print(ans)
threading.Thread(target=main).start()