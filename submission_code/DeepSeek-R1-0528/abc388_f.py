import sys
from collections import deque

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        print("No")
        return

    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = int(next(it))
    B = int(next(it))
    
    bad_intervals = []
    for _ in range(M):
        L = int(next(it))
        R = int(next(it))
        bad_intervals.append((L, R))
    
    bad_intervals.sort()
    
    segments = []
    current = 1
    for L, R in bad_intervals:
        if current < L:
            segments.append((current, L - 1))
        current = R + 1
    if current <= N:
        segments.append((current, N))
    
    n_seg = len(segments)
    if n_seg == 0:
        print("No")
        return

    visited = [False] * n_seg
    q = deque()
    q.append(0)
    visited[0] = True
    
    while q:
        i = q.popleft()
        s_i, t_i = segments[i]
        
        j = i + 1
        while j < n_seg:
            s_j, t_j = segments[j]
            if s_j < t_i + 1:
                j += 1
                continue
            if s_j > t_i + B:
                break
                
            L_gap = s_j - t_i
            R_gap = t_j - s_i
            
            if L_gap > B or R_gap < A:
                j += 1
                continue
                
            low_intersect = max(L_gap, A)
            high_intersect = min(R_gap, B)
            if low_intersect <= high_intersect and not visited[j]:
                visited[j] = True
                q.append(j)
                
            j += 1

    low_bound = max(1, N - B)
    high_bound = N - A
    found = False
    if low_bound <= high_bound:
        for i in range(n_seg):
            if visited[i]:
                s_i, t_i = segments[i]
                if not (t_i < low_bound or s_i > high_bound):
                    found = True
                    break
                    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()