import array
import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
    try:
        K = int(data[0].strip())
        S = data[1].strip()
        T = data[2].strip()
    except Exception:
        print("No")
        return

    n = len(S)
    m = len(T)
    
    if abs(n - m) > K:
        print("No")
        return

    width = 2 * K + 1
    dist = []
    big = 127
    for i in range(n + 1):
        dist.append(array.array('b', [big] * width))
    
    start_d = 0
    d_index_start = start_d + K
    if d_index_start < 0 or d_index_start >= width:
        print("No")
        return
    dist[0][d_index_start] = 0
    q = deque()
    q.append((0, start_d))
    
    while q:
        i, d = q.popleft()
        d_index = d + K
        cost = dist[i][d_index]
        j = i + d
        
        if i == n and j == m:
            if cost <= K:
                print("Yes")
                return
                
        if i < n and j < m and S[i] == T[j]:
            new_i = i + 1
            new_d = d
            new_d_index = new_d + K
            if -K <= new_d <= K:
                if cost < dist[new_i][new_d_index]:
                    dist[new_i][new_d_index] = cost
                    q.appendleft((new_i, new_d))
                    
        if j < m:
            new_i = i
            new_d = d + 1
            new_d_index = new_d + K
            if -K <= new_d <= K and cost + 1 <= K:
                if cost + 1 < dist[new_i][new_d_index]:
                    dist[new_i][new_d_index] = cost + 1
                    q.append((new_i, new_d))
                    
        if i < n:
            new_i = i + 1
            new_d = d - 1
            new_d_index = new_d + K
            if -K <= new_d <= K and cost + 1 <= K:
                if cost + 1 < dist[new_i][new_d_index]:
                    dist[new_i][new_d_index] = cost + 1
                    q.append((new_i, new_d))
                    
        if i < n and j < m and S[i] != T[j]:
            new_i = i + 1
            new_d = d
            new_d_index = new_d + K
            if -K <= new_d <= K and cost + 1 <= K:
                if cost + 1 < dist[new_i][new_d_index]:
                    dist[new_i][new_d_index] = cost + 1
                    q.append((new_i, new_d))
                    
    print("No")

if __name__ == "__main__":
    main()