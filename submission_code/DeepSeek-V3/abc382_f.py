# YOUR CODE HERE
import sys

def main():
    H, W, N = map(int, sys.stdin.readline().split())
    bars = []
    for _ in range(N):
        R, C, L = map(int, sys.stdin.readline().split())
        bars.append((R, C, L))
    
    # Initialize the final positions
    final_R = [0] * N
    # We need to process the bars in order of their initial row, from bottom to top
    # So we sort the bars by their initial row in descending order
    sorted_bars = sorted([(R, C, L, i) for i, (R, C, L) in enumerate(bars)], key=lambda x: -x[0])
    
    # To keep track of the occupied cells, we can use a dictionary or a list of sets
    # Since W can be up to 2e5, a list of sets is manageable
    occupied = [set() for _ in range(H+2)]
    
    for bar in sorted_bars:
        R, C, L, idx = bar
        # Determine the final row for this bar
        # Start from the initial row and move down until it can't move further
        current_R = R
        while current_R < H:
            # Check if all cells below are unoccupied
            can_move = True
            for j in range(C, C+L):
                if j in occupied[current_R + 1]:
                    can_move = False
                    break
            if not can_move:
                break
            current_R += 1
        # Update the final position
        final_R[idx] = current_R
        # Mark the cells as occupied
        for j in range(C, C+L):
            occupied[current_R].add(j)
    
    for i in range(N):
        print(final_R[i])

if __name__ == "__main__":
    main()