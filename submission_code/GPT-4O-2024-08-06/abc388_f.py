def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = int(data[2])
    B = int(data[3])
    
    bad_intervals = []
    index = 4
    for _ in range(M):
        L = int(data[index])
        R = int(data[index + 1])
        bad_intervals.append((L, R))
        index += 2
    
    # Create a set of bad squares
    bad_squares = set()
    for L, R in bad_intervals:
        for i in range(L, R + 1):
            bad_squares.add(i)
    
    # BFS to find if we can reach N
    from collections import deque
    
    queue = deque([1])
    visited = set([1])
    
    while queue:
        current = queue.popleft()
        
        # Try all possible moves from current
        for step in range(A, B + 1):
            next_square = current + step
            
            if next_square > N:
                continue
            
            if next_square == N:
                print("Yes")
                return
            
            if next_square not in bad_squares and next_square not in visited:
                visited.add(next_square)
                queue.append(next_square)
    
    print("No")