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
    # Sort bars by their initial row in ascending order
    # Since the order of processing is important
    # We need to process them in the order of their initial row
    # But since the bars are processed in order of their index, we need to sort them by their initial row
    # So that when we process them, the ones with lower initial row are processed first
    # But the problem says that the bars are processed in order of their index, not their initial row
    # So we need to process them in the order of their index, but for each step, we need to consider the current positions
    # So we need to simulate the process step by step
    
    # To simulate, we need to keep track of the current positions of all bars
    # We can represent the grid as a dictionary where the key is the row and the value is a list of occupied columns
    # But since H and W can be up to 2e5, we need a more efficient way
    # Instead, we can represent the occupied cells as a set of tuples (row, column)
    # But with N up to 2e5, this could be too slow
    # Alternatively, we can represent the occupied cells as a list of lists, where each row has a list of occupied columns
    # But with H up to 2e5, this is also not feasible
    # So we need a different approach
    
    # Since the bars are processed in order of their index, and each bar can only move down if the cells below are empty
    # We can process each bar in order, and for each bar, we can find the maximum row it can reach by checking the cells below it
    # We can use a list to keep track of the occupied columns for each row
    # Initialize a list of sets, one for each row
    occupied = [set() for _ in range(H+1)]
    for idx, (R, C, L) in enumerate(bars):
        for j in range(C, C+L):
            occupied[R].add(j)
    
    # Now, for each bar, we can find the maximum row it can reach
    for idx, (R, C, L) in enumerate(bars):
        current_R = R
        while True:
            if current_R == H:
                break
            # Check if all cells below are empty
            can_move = True
            for j in range(C, C+L):
                if (current_R + 1, j) in occupied[current_R + 1]:
                    can_move = False
                    break
            if not can_move:
                break
            # Move down
            current_R += 1
        final_R[idx] = current_R
    
    for r in final_R:
        print(r)

if __name__ == "__main__":
    main()