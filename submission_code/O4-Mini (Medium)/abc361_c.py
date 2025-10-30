import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # We need to choose N-K elements so that their max-min is minimized.
    # Since order doesn't affect max/min, we can sort and take the
    # smallest range among all windows of length N-K.
    M = N - K
    A.sort()

    best = 10**18
    # Slide a window of size M over the sorted array.
    for i in range(N - M + 1):
        diff = A[i + M - 1] - A[i]
        if diff < best:
            best = diff

    print(best)

if __name__ == "__main__":
    main()