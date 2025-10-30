import sys
import numpy as np

def main():
    import sys
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    input = sys.stdin.read().split()
    N = int(input[0])
    A = np.array(list(map(int, input[1:N+1])), dtype=np.int64)

    def is_possible(k):
        if 2*k-1 > N:
            return False
        req = np.minimum(np.arange(1, 2*k, dtype=np.int64), np.arange(2*k-1, 0, -1, dtype=np.int64))
        windows = sliding_window_view(A, window_shape=2*k-1)
        # Broadcast req to match windows shape
        # Compare all elements in windows >= req
        comparison = windows >= req
        # Check if any window has all True
        return np.any(np.all(comparison, axis=1))

    left =1
    right = (N +1)//2
    answer =1
    while left <= right:
        mid = (left + right)//2
        if is_possible(mid):
            answer = mid
            left = mid +1
        else:
            right = mid -1
    print(answer)

if __name__ == "__main__":
    main()