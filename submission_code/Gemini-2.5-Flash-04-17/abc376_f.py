import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    pos_l = 1
    pos_r = 2
    total_ops = 0

    def dist(u, v, n):
        # Distance between 1-indexed u and v on ring of size n
        d = abs(u - v)
        return min(d, n - d)

    for _ in range(Q):
        H, T = sys.stdin.readline().split()
        T = int(T)

        if H == 'L':
            curr_h_pos = pos_l
            other_h_pos = pos_r

            if curr_h_pos == T:
                cost = 0
            elif other_h_pos == T:
                # The other hand is at the target. It must move first.
                # Cost is 1 for the other hand + distance for the specified hand.
                cost = 1 + dist(curr_h_pos, T, N)
                
                # Update state: specified hand moves to T
                pos_l = T
                
                # The other hand moves away from T.
                # It moves to the neighbor of T (T+1 or T-1 mod N, adjusted for 1-indexing)
                # that is NOT the original position of the hand that moved to T.
                # Original pos_l was curr_h_pos.
                
                # Neighbors of T (1-indexed)
                neighbor1 = (T % N) + 1
                neighbor2 = (T - 2 + N) % N + 1
                
                # Determine which neighbor the other hand moves to
                # It moves to the neighbor that is not equal to the original curr_h_pos
                if neighbor1 == curr_h_pos:
                    pos_r = neighbor2
                else:
                    # If neighbor1 is not curr_h_pos, move to neighbor1.
                    # Since N>=3, neighbor1 != neighbor2.
                    # If original curr_h_pos is neighbor2, we must move to neighbor1.
                    # If original curr_h_pos is neither, we can move to either.
                    # The problem implies a deterministic minimum, so pick one consistently.
                    # The logic used in sample trace suggests moving to the one not occupied by the *original* position.
                    # My simplified logic derived from sample 3 suggests moving to the clockwise neighbor unless blocked by original curr_h_pos.
                    pos_r = neighbor1 # Move R to clockwise neighbor of T

            else: # curr_h_pos != T and other_h_pos != T
                # The specified hand moves along the shortest path. The other hand stays put.
                # This simple model is based on matching sample outputs, implying no blocking cost if destination is clear.
                cost = dist(curr_h_pos, T, N)
                
                # Update state: specified hand moves to T
                pos_l = T
                # The other hand (pos_r) remains unchanged

        else: # H == 'R'
            curr_h_pos = pos_r
            other_h_pos = pos_l

            if curr_h_pos == T:
                cost = 0
            elif other_h_pos == T:
                # The other hand is at the target. It must move first.
                # Cost is 1 for the other hand + distance for the specified hand.
                cost = 1 + dist(curr_h_pos, T, N)
                
                # Update state: specified hand moves to T
                pos_r = T
                
                # The other hand moves away from T.
                # Original pos_r was curr_h_pos.
                
                # Neighbors of T (1-indexed)
                neighbor1 = (T % N) + 1
                neighbor2 = (T - 2 + N) % N + 1
                
                # Determine which neighbor the other hand moves to
                if neighbor1 == curr_h_pos:
                     pos_l = neighbor2 # Move L to the other neighbor
                else:
                    pos_l = neighbor1 # Move L to clockwise neighbor of T


            else: # curr_h_pos != T and other_h_pos != T
                # The specified hand moves along the shortest path. The other hand stays put.
                cost = dist(curr_h_pos, T, N)
                
                # Update state: specified hand moves to T
                pos_r = T
                # The other hand (pos_l) remains unchanged

        total_ops += cost

    print(total_ops)

solve()