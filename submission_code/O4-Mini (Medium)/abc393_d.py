import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Collect 1-based positions of '1'
    positions = []
    for idx, ch in enumerate(S, start=1):
        if ch == '1':
            positions.append(idx)

    k = len(positions)
    # Compute q_i = positions[i] - (i+1), for i=0..k-1
    # Note: (i+1) is the 1-based index of the i-th one
    q = [positions[i] - (i + 1) for i in range(k)]

    # q is non-decreasing already; pick median
    mid = k // 2
    median_q = q[mid]

    # The minimal number of adjacent swaps is sum of |q_i - median_q|
    ans = 0
    for v in q:
        # absolute difference
        if v >= median_q:
            ans += v - median_q
        else:
            ans += median_q - v

    print(ans)

if __name__ == "__main__":
    main()