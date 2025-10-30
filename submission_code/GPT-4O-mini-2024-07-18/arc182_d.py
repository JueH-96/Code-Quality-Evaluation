def min_operations_to_make_good_sequence(N, M, A, B):
    total_operations = 0
    
    for i in range(N):
        if A[i] == B[i]:
            continue
        
        # Calculate the distance in both directions
        forward_distance = (B[i] - A[i]) % M
        backward_distance = (A[i] - B[i]) % M
        
        # Check if the change is valid
        if i > 0 and B[i] == A[i - 1]:
            # If B[i] is equal to A[i-1], we cannot set A[i] to B[i] directly
            if forward_distance < backward_distance:
                # We need to go backward
                total_operations += backward_distance
            else:
                # We need to go forward
                total_operations += forward_distance
        elif i < N - 1 and B[i] == A[i + 1]:
            # If B[i] is equal to A[i+1], we cannot set A[i] to B[i] directly
            if backward_distance < forward_distance:
                # We need to go forward
                total_operations += forward_distance
            else:
                # We need to go backward
                total_operations += backward_distance
        else:
            # We can set A[i] to B[i] directly
            total_operations += min(forward_distance, backward_distance)
    
    return total_operations

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N + 2]))
B = list(map(int, data[N + 2:2 * N + 2]))

result = min_operations_to_make_good_sequence(N, M, A, B)
print(result)