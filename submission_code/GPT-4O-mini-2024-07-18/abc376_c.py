def minimum_box_size(N, A, B):
    # Sort the toy sizes and box sizes
    A.sort()
    B.sort()
    
    # We need to find the minimum x such that we can place all toys in boxes
    # We will use a greedy approach to try to fit the toys into boxes
    
    # Create a list of all box sizes including the new box size x
    boxes = B + [0]  # We will determine the size of the new box later
    
    # We will try to fit each toy into the boxes
    j = 0  # Pointer for boxes
    for toy in A:
        # Find the first box that can fit this toy
        while j < len(boxes) and boxes[j] < toy:
            j += 1
        if j == len(boxes):
            # No box can fit this toy, we need to increase x
            return -1
    
        # Place the toy in the box
        boxes[j] = -1  # Mark this box as used
        j += 1  # Move to the next box
    
    # Now we need to determine the minimum size of the new box x
    # The new box must be at least as large as the largest toy that couldn't be placed
    # which is the last toy in A since we placed all toys in boxes
    return max(A) if j == len(boxes) else -1

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N-1]))
    
    result = minimum_box_size(N, A, B)
    print(result)

if __name__ == "__main__":
    main()