import sys

def main():
    input = sys.stdin.read
    N, M, *AB = map(int, input().split())
    A = AB[:N]
    B = AB[N:]
    
    # Initialize the boxes
    boxes = [0] * N
    
    # Perform the operations
    for b in B:
        balls = A[b]
        A[b] = 0
        for i in range(balls):
            boxes[(b + i + 1) % N] += 1
    
    # Calculate the final number of balls in each box
    for i in range(N):
        boxes[i] += A[i]
    
    # Output the result
    print(' '.join(map(str, boxes)))

if __name__ == "__main__":
    main()