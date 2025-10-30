# YOUR CODE HERE
def find_min_x(N, A, B):
    # Sort the toys and boxes
    A.sort()
    B.sort()
    
    # Binary search for the minimum x
    left, right = 1, 10**9
    answer = -1
    
    while left <= right:
        mid = (left + right) // 2
        # Create a new list of boxes including the new box of size mid
        boxes = B + [mid]
        boxes.sort()
        
        # Check if we can place all toys in the boxes
        possible = True
        for i in range(N):
            if A[i] > boxes[i]:
                possible = False
                break
        
        if possible:
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    
    return answer

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:2*N]))

print(find_min_x(N, A, B))