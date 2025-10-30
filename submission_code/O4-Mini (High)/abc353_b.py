def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    empty = K
    starts = 0
    idx = 0
    while idx < N:
        if A[idx] > empty:
            # Not enough seats: start the ride
            starts += 1
            empty = K
        else:
            # Guide this group in
            empty -= A[idx]
            idx += 1
    # Finally, start the ride for the remaining guided groups (or if queue empty)
    starts += 1
    
    print(starts)

if __name__ == "__main__":
    main()