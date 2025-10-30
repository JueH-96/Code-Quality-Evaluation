# YOUR CODE HERE
import sys
from collections import deque

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    # dragon_coords stores the (x,y) positions of the dragon parts that are currently
    # managed dynamically, starting from the head.
    # dragon_coords[0] is the head (part 1).
    # dragon_coords[k] is part k+1.
    # The maximum length of this deque will be N.
    # If len(dragon_coords) < N, it means parts from len(dragon_coords)+1 to N are still
    # at their initial (i,0) positions.
    dragon_coords = deque()

    # Initialize the head's position. Part 1 starts at (1,0).
    # Other parts (2 to N) are initially at (i,0).
    dragon_coords.append((1, 0))

    for _ in range(Q):
        query = sys.stdin.readline().split()
        query_type = int(query[0])

        if query_type == 1: # Move query
            direction = query[1]
            
            # Get current head position from the front of the deque
            current_head_x, current_head_y = dragon_coords[0]

            # Calculate new head position based on direction
            new_head_x, new_head_y = current_head_x, current_head_y
            if direction == 'R':
                new_head_x += 1
            elif direction == 'L':
                new_head_x -= 1
            elif direction == 'U':
                new_head_y += 1
            elif direction == 'D':
                new_head_y -= 1
            
            # Add the new head position to the front of the deque.
            # This naturally pushes the previous head position to index 1,
            # which correctly becomes the new position for part 2, and so on.
            dragon_coords.appendleft((new_head_x, new_head_y))

            # If the deque's length exceeds N, it means the N-th part has moved,
            # and the part that was at the very end of the deque (representing
            # where the N-th part *used to be*) is now "off the dragon".
            # So, we remove the last element to keep the deque's size at most N.
            if len(dragon_coords) > N:
                dragon_coords.pop()

        elif query_type == 2: # Query position query
            p = int(query[1])
            
            # If part 'p' is among the dynamically tracked parts (i.e., its position
            # has been affected by previous moves of the head).
            # Note: `p` is 1-indexed, so we access `dragon_coords[p - 1]`.
            if p <= len(dragon_coords):
                x, y = dragon_coords[p - 1]
                sys.stdout.write(f"{x} {y}
")
            else:
                # If part 'p' is beyond the dynamically tracked parts, it means
                # the "wave" of movement hasn't reached it yet.
                # It's still in its initial position (p, 0).
                sys.stdout.write(f"{p} 0
")

solve()