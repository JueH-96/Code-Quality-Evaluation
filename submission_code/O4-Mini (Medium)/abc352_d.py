import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = [int(next(it)) for _ in range(N)]
    # pos[v] = index of value v in P (1-based positions)
    pos = [0]*(N+1)
    for idx, v in enumerate(P, start=1):
        pos[v] = idx
    # build array of positions for values 1..N
    arr = pos[1:]  # length N, 0-based

    from collections import deque
    min_dq = deque()  # will store indices in arr window, increasing arr values
    max_dq = deque()  # will store indices, decreasing arr values

    ans = N  # max possible is N-1
    for i in range(N):
        # push arr[i] into min_dq
        while min_dq and arr[i] <= arr[min_dq[-1]]:
            min_dq.pop()
        min_dq.append(i)
        # push into max_dq
        while max_dq and arr[i] >= arr[max_dq[-1]]:
            max_dq.pop()
        max_dq.append(i)

        # when we have a full window of size K
        if i >= K-1:
            start = i - (K-1)
            # pop out-of-window indices
            if min_dq[0] < start:
                min_dq.popleft()
            if max_dq[0] < start:
                max_dq.popleft()
            # current min and max in window
            cur_min = arr[min_dq[0]]
            cur_max = arr[max_dq[0]]
            diff = cur_max - cur_min
            if diff < ans:
                ans = diff

    # print the minimal i_K - i_1
    print(ans)

if __name__ == "__main__":
    main()