import sys
from collections import deque, defaultdict
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
        
    it = iter(data)
    N = int(next(it)); M = int(next(it)); A = int(next(it)); B = int(next(it))
    bad_intervals = []
    for _ in range(M):
        L = int(next(it)); R = int(next(it))
        bad_intervals.append((L, R))
    
    segments = []
    current_start = 1
    for L, R in bad_intervals:
        if current_start < L:
            segments.append((current_start, L-1))
        current_start = R + 1
    if current_start <= N:
        segments.append((current_start, N))
    
    seg_starts = [s for s, e in segments]
    n_seg = len(segments)
    
    entries = defaultdict(set)
    entries[0].add(1)
    
    found = False
    for i in range(n_seg):
        s_i, e_i = segments[i]
        window_start = e_i + 1 - 2 * B
        visited = set()
        queue = deque()
        exit_points = set()
        
        for x in entries[i]:
            if x > e_i:
                continue
            if x >= window_start:
                if x not in visited:
                    visited.add(x)
                    queue.append(x)
            else:
                if e_i - x >= (A - 1) * B:
                    start_pos = max(s_i, window_start)
                    for pos in range(start_pos, e_i + 1):
                        if pos not in visited:
                            visited.add(pos)
                            queue.append(pos)
                else:
                    q_local = deque()
                    visited_local = set()
                    q_local.append(x)
                    visited_local.add(x)
                    while q_local:
                        y = q_local.popleft()
                        if y >= window_start and y <= e_i:
                            if y not in visited:
                                visited.add(y)
                                queue.append(y)
                        for step in range(A, B + 1):
                            z = y + step
                            if z > e_i:
                                break
                            if z not in visited_local:
                                visited_local.add(z)
                                q_local.append(z)
                                
        while queue:
            y = queue.popleft()
            if y == N:
                print("Yes")
                found = True
                break
                
            if y >= e_i + 1 - B:
                exit_points.add(y)
                
            for step in range(A, B + 1):
                z = y + step
                if z > e_i:
                    break
                if z < window_start:
                    continue
                if z not in visited:
                    visited.add(z)
                    queue.append(z)
                    
        if found:
            break
            
        for y in exit_points:
            for step in range(A, B + 1):
                z = y + step
                if z > N:
                    continue
                if z == N:
                    print("Yes")
                    found = True
                    break
                idx = bisect.bisect_right(seg_starts, z) - 1
                if idx < 0 or idx >= n_seg:
                    continue
                s_j, e_j = segments[idx]
                if s_j <= z <= e_j:
                    if z not in entries[idx]:
                        entries[idx].add(z)
            if found:
                break
                
        if found:
            break
            
    print("No" if not found else "")
    
if __name__ == "__main__":
    main()