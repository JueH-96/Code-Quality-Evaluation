# YOUR CODE HERE
import sys
import math
import itertools

def main():
    import sys
    import math
    from itertools import permutations, product

    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    S = int(input[idx]); idx +=1
    T = int(input[idx]); idx +=1

    lines = []
    for _ in range(N):
        A = int(input[idx]); idx +=1
        B = int(input[idx]); idx +=1
        C = int(input[idx]); idx +=1
        D = int(input[idx]); idx +=1
        lines.append( ((A,B), (C,D)) )

    min_time = float('inf')

    for perm in permutations(lines):
        # Each perm is a tuple of N lines
        for directions in product([0,1], repeat=N):
            time = 0.0
            pos = (0.0, 0.0)
            for line, dir_choice in zip(perm, directions):
                if dir_choice == 0:
                    start_p, end_p = line[0], line[1]
                else:
                    start_p, end_p = line[1], line[0]
                # Move to start_p
                move_dist = math.hypot(start_p[0] - pos[0], start_p[1] - pos[1])
                move_time = move_dist / S
                time += move_time
                # Emit laser along the segment
                emit_dist = math.hypot(end_p[0] - start_p[0], end_p[1] - start_p[1])
                emit_time = emit_dist / T
                time += emit_time
                # Update position
                pos = end_p
                # Early pruning if time already exceeds current min_time
                if time >= min_time:
                    break
            if time < min_time:
                min_time = time

    # Print with enough precision
    print("{0:.20f}".format(min_time))

if __name__ == "__main__":
    main()