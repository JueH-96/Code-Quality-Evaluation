import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Check if we can form k kagamimochi pairs
    def can_make(k):
        # Use the smallest k as tops, largest k as bottoms
        # small A[i], large A[N-k+i]
        for i in range(k):
            if A[i] * 2 > A[N - k + i]:
                return False
        return True

    # Binary search for the maximum k in [0, N//2]
    low, high = 0, N // 2
    while low < high:
        mid = (low + high + 1) // 2
        if can_make(mid):
            low = mid
        else:
            high = mid - 1

    print(low)

if __name__ == "__main__":
    main()