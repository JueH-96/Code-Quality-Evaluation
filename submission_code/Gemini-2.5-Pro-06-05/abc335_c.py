# YOUR CODE HERE
import sys

def main():
    """
    Solves the dragon game problem by tracking the head's path.
    """
    # Read problem parameters. N is not used in the optimized logic.
    N, Q = map(int, sys.stdin.readline().split())

    # head_path stores the coordinates of the head (part 1) over time.
    # The index `t` corresponds to the state after `t` moves.
    # At time t=0 (initially), the head is at (1, 0).
    head_path = [(1, 0)]

    for _ in range(Q):
        # Read each query
        query_type, val = sys.stdin.readline().split()
        query_type = int(query_type)

        if query_type == 1:
            # Move query: "1 C"
            direction = val
            
            # Get the current position of the head.
            last_x, last_y = head_path[-1]
            
            # Calculate and record the new head position.
            if direction == 'R':
                head_path.append((last_x + 1, last_y))
            elif direction == 'L':
                head_path.append((last_x - 1, last_y))
            elif direction == 'U':
                head_path.append((last_x, last_y + 1))
            else:  # direction == 'D'
                head_path.append((last_x, last_y - 1))
        
        else:  # query_type == 2
            # Position query: "2 p"
            p = int(val)
            
            # The current time `t` is the number of moves made so far.
            time = len(head_path) - 1

            # Determine if part p's position is on the recorded path or initial path.
            if p <= time + 1:
                # Part p's position corresponds to a recorded head position.
                # Its position at `time` is the head's position at `time - p + 1`.
                idx = time - p + 1
                # The '*' unpacks the tuple into arguments for print.
                print(*head_path[idx])
            else:
                # Part p is still on its initial linear path.
                # Its position is (p - time, 0).
                x = p - time
                print(x, 0)

if __name__ == "__main__":
    main()