import sys
from collections import deque

def solve():
    N, M, A, B = map(int, sys.stdin.readline().split())
    segments = []
    for _ in range(M):
        L, R = map(int, sys.stdin.readline().split())
        segments.append((L, R))
    
    # The bad segments are sorted in increasing order as per the problem statement
    # So no need to sort again
    
    q = deque()
    q.append(1)
    visited = set()
    visited.add(1)
    found = False
    
    while q:
        current = q.popleft()
        # Generate all possible next steps: current + i where A <= i <= B
        for i in range(A, B + 1):
            next_pos = current + i
            if next_pos > N:
                continue
            # Check if next_pos is in any bad segment
            bad = False
            # Since segments are sorted, use binary search to find overlapping segments
            low = 0
            high = M - 1
            while low <= high:
                mid = (low + high) // 2
                L, R = segments[mid]
                if R < next_pos:
                    low = mid + 1
                elif L > next_pos:
                    high = mid - 1
                else:
                    bad = True
                    break
            if not bad:
                if next_pos == N:
                    found = True
                    break
                if next_pos not in visited:
                    visited.add(next_pos)
                    q.append(next_pos)
        if found:
            break
    print("Yes" if found else "No")

solve()