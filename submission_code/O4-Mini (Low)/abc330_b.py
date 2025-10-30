import sys
import threading

def main():
    data = sys.stdin.read().split()
    # Parse input
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    A = list(map(int, data[3:]))

    # For each A_i, clamp it to [L, R]
    res = []
    for a in A:
        if a < L:
            res.append(L)
        elif a > R:
            res.append(R)
        else:
            res.append(a)

    # Print results
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()