import itertools
import math

def main():
    n, S, T = map(int, input().split())
    lines = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        dir1 = ((a, b), (c, d))
        dir2 = ((c, d), (a, b))
        lines.append((dir1, dir2))
    
    min_time = float('inf')
    
    # Iterate over all permutations of the lines
    for perm in itertools.permutations(range(n)):
        # Iterate over all possible direction choices for each line in the permutation
        for directions in itertools.product([0, 1], repeat=n):
            current_pos = (0.0, 0.0)
            total_time = 0.0
            for i, line_idx in enumerate(perm):
                # Get the direction for the current line in the permutation
                dir_choice = directions[i]
                start, end = lines[line_idx][dir_choice]
                
                # Calculate movement time from current position to start point
                dx = start[0] - current_pos[0]
                dy = start[1] - current_pos[1]
                dist = math.hypot(dx, dy)
                total_time += dist / S
                
                # Calculate emitting time for the line segment
                line_length = math.hypot(end[0] - start[0], end[1] - start[1])
                total_time += line_length / T
                
                # Update current position to the end of the line
                current_pos = (end[0], end[1])
            
            # Update the minimum time found
            if total_time < min_time:
                min_time = total_time
    
    # Output with sufficient precision
    print("{0:.20f}".format(min_time))

if __name__ == "__main__":
    main()