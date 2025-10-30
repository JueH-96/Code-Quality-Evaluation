# YOUR CODE HERE
import sys
input = sys.stdin.read

def distribute_balls(N, M, A, B):
    # Initialize the boxes with the initial number of balls
    boxes = A[:]
    
    for i in range(M):
        box_index = B[i]
        balls_in_hand = boxes[box_index]
        boxes[box_index] = 0
        
        # Distribute the balls
        for j in range(balls_in_hand):
            target_box = (box_index + j + 1) % N
            boxes[target_box] += 1
    
    return boxes

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    result = distribute_balls(N, M, A, B)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()