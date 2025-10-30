def calculate_expression(N, A):
    total_sum = 0
    suffix_sum = 0
    count = 0
    
    # Traverse the array from the end to the beginning
    for i in range(N - 1, -1, -1):
        # For each A[i], we add the contribution of all A[j] where j > i
        total_sum += suffix_sum - count * A[i]
        
        # Update suffix sum and count
        suffix_sum += A[i]
        count += 1
    
    return total_sum

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N + 1]))

result = calculate_expression(N, A)
print(result)