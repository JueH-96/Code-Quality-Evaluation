import sys
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    
    pos = [0] * (N + 1)  # pos[1] to pos[N]
    for idx in range(N):
        v = P[idx]
        pos[v] = idx + 1  # 1-based position
    
    arr = [pos[v] for v in range(1, N + 1)]
    
    def get_sliding_min(arr, k):
        q = deque()
        result = []
        for i in range(len(arr)):
            # Remove elements >= current from the end
            while q and arr[q[-1]] >= arr[i]:
                q.pop()
            q.append(i)
            # Remove elements outside the window
            while q[0] < i - k + 1:
                q.popleft()
            if i >= k - 1:
                result.append(arr[q[0]])
        return result
    
    def get_sliding_max(arr, k):
        q = deque()
        result = []
        for i in range(len(arr)):
            # Remove elements <= current from the end
            while q and arr[q[-1]] <= arr[i]:
                q.pop()
            q.append(i)
            # Remove elements outside the window
            while q[0] < i - k + 1:
                q.popleft()
            if i >= k - 1:
                result.append(arr[q[0]])
        return result
    
    min_win = get_sliding_min(arr, K)
    max_win = get_sliding_max(arr, K)
    
    min_diff = float('inf')
    for m, M in zip(min_win, max_win):
        diff = M - m
        if diff < min_diff:
            min_diff = diff
    
    print(min_diff)

if __name__ == "__main__":
    main()