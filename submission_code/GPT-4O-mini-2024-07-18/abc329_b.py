# YOUR CODE HERE
def find_second_largest(N, A):
    largest = max(A)
    second_largest = -1
    
    for number in A:
        if number != largest and number > second_largest:
            second_largest = number
            
    return second_largest

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

result = find_second_largest(N, A)
print(result)