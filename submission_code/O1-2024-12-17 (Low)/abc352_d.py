def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    P = list(map(int, input_data[2:]))

    # loc[x] = index i where P[i] = x (1-based index).
    # We'll store these 1-based for clarity, then build an array L of length N:
    # L[x-1] = loc[x] for x=1..N.
    loc = [0] * (N+1)
    for i, val in enumerate(P, start=1):
        loc[val] = i

    # Build array L of length N: L[i] = loc[i+1] in 1-based indexing
    L = [loc[i+1] for i in range(N)]

    # We'll compute the sliding window minima and maxima for windows of size K over L.

    from collections import deque

    def sliding_window_min(arr, k):
        """Returns an array of length len(arr)-k+1 where
           result[i] = min(arr[i..i+k-1])"""
        dq = deque()
        result = []
        for i, x in enumerate(arr):
            # Pop from back while last element >= x
            while dq and arr[dq[-1]] >= x:
                dq.pop()
            dq.append(i)
            # Remove front if out of window
            if i >= k and dq[0] == i - k:
                dq.popleft()
            # i-k+1 is the start of the window
            if i >= k - 1:
                result.append(arr[dq[0]])
        return result

    def sliding_window_max(arr, k):
        """Returns an array of length len(arr)-k+1 where
           result[i] = max(arr[i..i+k-1])"""
        dq = deque()
        result = []
        for i, x in enumerate(arr):
            # Pop from back while last element <= x
            while dq and arr[dq[-1]] <= x:
                dq.pop()
            dq.append(i)
            # Remove front if out of window
            if i >= k and dq[0] == i - k:
                dq.popleft()
            if i >= k - 1:
                result.append(arr[dq[0]])
        return result

    min_arr = sliding_window_min(L, K)
    max_arr = sliding_window_max(L, K)

    # Now find the minimum difference between max and min for each window
    answer = float('inf')
    for i in range(len(min_arr)):
        diff = max_arr[i] - min_arr[i]
        if diff < answer:
            answer = diff

    print(answer)

# Do not forget to call main()
main()