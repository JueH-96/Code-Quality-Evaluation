# YOUR CODE HERE
import sys
import threading
def main():
    import bisect
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    idx = 1
    A = list(map(int, N_and_rest[idx:idx+N]))
    idx += N
    B = list(map(int, N_and_rest[idx:idx+N-1]))
    idx += N -1

    N_toys = N
    N_boxes = N -1
    toy_sizes = A
    box_sizes = B

    def is_possible(x):
        # We have N boxes: N-1 existing boxes + new box of size x
        boxes = box_sizes + [x]
        boxes.sort()
        toys = sorted(toy_sizes)
        box_index = 0
        N_boxes = N  # After adding new box
        N_toys = N
        for toy in toys:
            # Find the first box that can hold the toy
            while box_index < N_boxes and boxes[box_index] < toy:
                box_index += 1
            if box_index == N_boxes:
                # No box can hold this toy
                return False
            # Assign toy to boxes[box_index]
            box_index +=1
        return True
        
    # Binary search for minimal x
    left = 1
    right = 2 * (10**9)
    answer = -1
    while left <= right:
        mid = (left + right) // 2
        if is_possible(mid):
            answer = mid
            right = mid -1
        else:
            left = mid + 1
    print(answer)
    

threading.Thread(target=main).start()