def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    P = list(map(int, data[2:]))

    # pos[x] = position of x in the permutation P (1-based).
    # For convenience, we'll build pos so that pos[v] = index of value v in P.
    pos = [0]*(N+1)
    for i, val in enumerate(P):
        pos[val] = i+1  # store 1-based positions

    # We'll create an array arr[] of length N where:
    # arr[x-1] = position of the value x in P
    arr = pos[1:]  # ignore pos[0], as values in P range from 1..N

    # If K = 1, any single element forms a valid subsequence (a..a), so answer = 0.
    if K == 1:
        print(0)
        return

    # We'll use a standard O(N) sliding window approach to compute
    # minimum and maximum in every window of size K in arr.

    def sliding_window_min(array, k):
        dq = deque()
        result = [0]*(len(array)-k+1)
        for i, val in enumerate(array):
            # Pop larger values from the back
            while dq and array[dq[-1]] >= val:
                dq.pop()
            dq.append(i)
            # Pop from the front if it's out of the window
            if dq[0] <= i - k:
                dq.popleft()
            # Once we've processed k elements, record the min
            if i >= k-1:
                result[i-k+1] = array[dq[0]]
        return result

    def sliding_window_max(array, k):
        dq = deque()
        result = [0]*(len(array)-k+1)
        for i, val in enumerate(array):
            # Pop smaller values from the back
            while dq and array[dq[-1]] <= val:
                dq.pop()
            dq.append(i)
            # Pop from the front if it's out of the window
            if dq[0] <= i - k:
                dq.popleft()
            # Once we've processed k elements, record the max
            if i >= k-1:
                result[i-k+1] = array[dq[0]]
        return result

    mins = sliding_window_min(arr, K)
    maxs = sliding_window_max(arr, K)

    answer = float('inf')
    for i in range(N-K+1):
        answer = min(answer, maxs[i] - mins[i])

    print(answer)

# Don't forget to call main()
if __name__ == "__main__":
    main()