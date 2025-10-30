import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, M = data[0], data[1]
    A = data[2:]
    
    A.sort()
    
    max_gifts = 0
    j = 0                      # right end of the sliding window (exclusive)
    
    for i in range(N):         # left end of the window
        # extend j while gifts stay inside [A[i], A[i] + M)
        while j < N and A[j] < A[i] + M:
            j += 1
        # all indices from i (inclusive) to j (exclusive) are inside the interval
        max_gifts = max(max_gifts, j - i)
        # when i moves right by one, j never needs to move back (monotone)
    
    print(max_gifts)

if __name__ == "__main__":
    main()