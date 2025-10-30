import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # We need to choose N-K elements so that the difference between
    # the maximum and minimum of the chosen elements is minimized.
    # This is equivalent to sorting the array and taking the minimum
    # range over all windows of size M = N-K.
    M = N - K
    A.sort()
    ans = 10**18
    # Slide a window of length M over the sorted array
    for i in range(N - M + 1):
        diff = A[i + M - 1] - A[i]
        if diff < ans:
            ans = diff

    print(ans)

if __name__ == "__main__":
    main()