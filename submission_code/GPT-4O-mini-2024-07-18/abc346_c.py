def sum_of_missing_numbers(N, K, A):
    # Convert A to a set for O(1) average time complexity on lookups
    A_set = set(A)
    
    # Initialize the sum of missing numbers
    missing_sum = 0
    
    # Iterate through numbers from 1 to K
    for i in range(1, K + 1):
        if i not in A_set:
            missing_sum += i
            
    return missing_sum

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:N+2]))

result = sum_of_missing_numbers(N, K, A)
print(result)