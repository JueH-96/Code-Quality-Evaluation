def solve(N, M, A, B):
    boxes = A.copy()
    
    for i in range(M):
        box_idx = B[i]
        balls_in_hand = boxes[box_idx]
        boxes[box_idx] = 0
        
        for d in range(N):
            remainder = (d - box_idx) % N
            if remainder == 0:
                # For box d = box_idx, we need j to be a multiple of N
                # These are balls with j = N, 2N, 3N, ...
                count = balls_in_hand // N
            else:
                # For other boxes, we need j % N = remainder
                # These are balls with j = remainder, remainder+N, remainder+2N, ...
                if remainder <= balls_in_hand:
                    count = (balls_in_hand - remainder) // N + 1
                else:
                    count = 0
            
            boxes[d] += count
    
    return boxes

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Solve and print output
result = solve(N, M, A, B)
print(" ".join(map(str, result)))