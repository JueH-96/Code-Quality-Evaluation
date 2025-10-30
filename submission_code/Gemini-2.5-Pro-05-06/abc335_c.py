import sys

def main():
    # N is the number of parts, Q is the number of queries.
    # N is not strictly needed for the algorithm's logic as p is guaranteed to be <= N.
    _N, Q = map(int, sys.stdin.readline().split())

    # head_history[k] stores the coordinates of the head (part 1) AFTER k moves.
    # Initially, after 0 moves, part 1 (head) is at (1,0).
    # So, head_history[0] = (1,0).
    head_history = [(1, 0)] 

    for _i_query in range(Q): # Loop Q times for Q queries
        query_line = sys.stdin.readline().split()
        query_type = int(query_line[0])

        if query_type == 1:
            # This is a "move" query: "1 C"
            direction = query_line[1]
            
            # The head moves from its last recorded position.
            current_head_x, current_head_y = head_history[-1]
            
            new_head_x, new_head_y = current_head_x, current_head_y
            
            if direction == 'R': # Positive x-direction
                new_head_x += 1
            elif direction == 'L': # Negative x-direction
                new_head_x -= 1
            elif direction == 'U': # Positive y-direction
                new_head_y += 1
            elif direction == 'D': # Negative y-direction
                new_head_y -= 1
            
            # Record the new head position in history.
            head_history.append((new_head_x, new_head_y))
        
        elif query_type == 2:
            # This is a "find position" query: "2 p"
            p = int(query_line[1]) # Part number to find
            
            # num_total_moves is the total number of type 1 (move) queries processed so far.
            # This is equivalent to k in the P_p^(k) notation.
            # len(head_history) = (number of moves made) + 1 (for initial position H(0)).
            # So, num_total_moves = len(head_history) - 1.
            num_total_moves = len(head_history) - 1
            
            # The core idea: The position of part p after k moves, P_p^(k),
            # is the same as the position of the head after k-(p-1) moves.
            # Let target_head_move_count = k-(p-1). This is the "effective" historical
            # move count for the head that determines part p's current position.
            target_head_move_count = num_total_moves - (p - 1)
            
            if target_head_move_count >= 0:
                # If target_head_move_count is non-negative, it means part p is at a location
                # that the head (part 1) occupied at or after its initial state (move 0).
                # This position is stored in head_history[target_head_move_count].
                ans_x, ans_y = head_history[target_head_move_count]
            else:
                # If target_head_move_count is negative, it means part p has not "yet"
                # reached any position that the head (part 1) occupied at or after move 0.
                # It's still effectively on its initial trajectory along the x-axis.
                # Its x-coordinate is p - num_total_moves. The y-coordinate is 0.
                ans_x = p - num_total_moves
                ans_y = 0
            
            # Print the coordinates for this query.
            # Using f-string for formatted output. print() adds a newline by default.
            print(f"{ans_x} {ans_y}")

if __name__ == '__main__':
    main()