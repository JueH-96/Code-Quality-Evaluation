import sys
import threading

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    prefix = 0
    min_prefix = 0
    total = 0

    # Track running prefix sum and its minimum
    for a in A:
        prefix += a
        if prefix < min_prefix:
            min_prefix = prefix
        total += a

    # We need initial x >= -min_prefix to keep all intermediates >= 0
    x = max(0, -min_prefix)
    # Current passengers = x + total sum of A
    print(x + total)

if __name__ == "__main__":
    main()