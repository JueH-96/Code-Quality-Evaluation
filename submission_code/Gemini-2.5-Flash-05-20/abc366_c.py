import sys

def solve():
    Q = int(sys.stdin.readline())

    # ball_counts: A dictionary to store the count of each integer ball.
    # Key: integer written on the ball
    # Value: number of balls with that integer currently in the bag
    ball_counts = {} 
    
    # unique_ball_types: An integer to store the number of different integers
    # currently present on balls in the bag.
    unique_ball_types = 0 

    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        query_type = query[0]

        if query_type == 1:
            # Type 1 query: Add a ball with integer x
            x = query[1]
            
            # Get the current count of balls with integer x.
            # If x is not yet in the dictionary, its count is 0.
            current_count = ball_counts.get(x, 0)
            
            # If the count was 0, it means this is the first ball of type x
            # being added, so the number of unique types increases.
            if current_count == 0:
                unique_ball_types += 1
            
            # Increment the count for integer x.
            ball_counts[x] = current_count + 1
            
        elif query_type == 2:
            # Type 2 query: Remove one ball with integer x
            x = query[1]
            
            # Decrement the count for integer x.
            # The problem guarantees that a ball with x is present,
            # so ball_counts[x] will be > 0.
            ball_counts[x] -= 1
            
            # If the count becomes 0, it means the last ball of type x
            # has been removed, so the number of unique types decreases.
            if ball_counts[x] == 0:
                unique_ball_types -= 1
                # Optional: Remove the key from the dictionary to save memory.
                # This doesn't affect correctness because .get(x, 0) handles
                # missing keys, but it can be beneficial for memory efficiency.
                del ball_counts[x]
                
        else: # query_type == 3
            # Type 3 query: Print the number of different integers
            # Simply print the current value of unique_ball_types.
            sys.stdout.write(str(unique_ball_types) + "
")

# Call the solve function to run the program
solve()