def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # We need to choose m = N-K numbers from A such that the difference between 
    # the maximum and the minimum of these chosen numbers is minimized.
    # Note that the order of elements in the remaining sequence B is the same as 
    # in the original array, but that constraint has no effect on the 
    # min and max operation.
    # This means we can choose any subset of size m. 
    # Therefore, the optimal choice is to take m numbers that are as close 
    # in value as possible. In other words, if we sort A, we want to find a window 
    # of length m such that A_sorted[i+m-1] - A_sorted[i] is minimized.
    
    m = N - K
    A.sort()
    best = float("inf")
    
    for i in range(0, N - m + 1):
        diff = A[i + m - 1] - A[i]
        if diff < best:
            best = diff
    
    sys.stdout.write(str(best))


if __name__ == '__main__':
    main()