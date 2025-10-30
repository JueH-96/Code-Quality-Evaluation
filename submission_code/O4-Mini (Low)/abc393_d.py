import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()
    # Collect positions of '1's (0-based)
    positions = []
    for i, ch in enumerate(S):
        if ch == '1':
            positions.append(i)
    k = len(positions)
    # If already contiguous (single 1 or all packed), cost is zero
    if k <= 1:
        print(0)
        return
    # Compute b[i] = positions[i] - i
    # b will be non-decreasing since positions is strictly increasing
    b = [pos - i for i, pos in enumerate(positions)]
    # Median of b
    m = b[k // 2]
    # Total cost is sum of |b[i] - m|
    total = 0
    for x in b:
        total += abs(x - m)
    print(total)

if __name__ == "__main__":
    main()