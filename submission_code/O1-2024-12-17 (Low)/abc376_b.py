from collections import deque

def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    
    # Instructions are in the format: H_i T_i
    # Example: R 4 or L 5
    # We'll parse them from input_data.
    instructions = []
    idx = 2
    for _ in range(Q):
        H = input_data[idx]
        T = int(input_data[idx+1])
        idx += 2
        instructions.append((H, T))
    
    # We'll keep track of the current positions of the hands:
    # left_hand starts at 1, right_hand starts at 2 (given).
    left_hand = 1
    right_hand = 2
    
    # Function to compute the shortest distance on a ring from start to end
    # with 'blocked' node that cannot be stepped on.
    def bfs_ring(start, end, blocked):
        # If start == end, distance is 0, but from the problem's guarantee T != other_hand.
        # We'll do a BFS on the ring ignoring 'blocked'.
        visited = [False]*(N+1)
        queue = deque([(start, 0)])
        visited[start] = True
        
        while queue:
            pos, dist = queue.popleft()
            if pos == end:
                return dist
            
            # neighbors on the ring:
            nxt1 = pos + 1
            nxt2 = pos - 1
            if nxt1 > N:
                nxt1 = 1
            if nxt2 < 1:
                nxt2 = N
            
            for nxt in (nxt1, nxt2):
                if nxt != blocked and not visited[nxt]:
                    visited[nxt] = True
                    queue.append((nxt, dist+1))
        
        return 0  # Should not happen for valid instructions
    
    total_moves = 0
    
    # Process each instruction
    for H, T in instructions:
        if H == 'L':
            # Move left hand from left_hand to T, blocking right_hand
            dist = bfs_ring(left_hand, T, right_hand)
            total_moves += dist
            left_hand = T
        else: # H == 'R'
            # Move right hand from right_hand to T, blocking left_hand
            dist = bfs_ring(right_hand, T, left_hand)
            total_moves += dist
            right_hand = T
    
    print(total_moves)

# Do not forget to call main()!
if __name__ == "__main__":
    main()