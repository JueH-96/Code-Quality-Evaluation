import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    A = list(map(int, data[3:]))

    # For each A_i, clamp it to the range [L, R].
    # That gives the unique X_i minimizing |X_i - A_i| over X_i in [L, R].
    result = []
    for a in A:
        if a < L:
            result.append(L)
        elif a > R:
            result.append(R)
        else:
            result.append(a)

    # Output space-separated
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()