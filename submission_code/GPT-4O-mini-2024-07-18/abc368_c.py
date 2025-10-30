# YOUR CODE HERE
def calculate_time_to_defeat_enemies(N, healths):
    T = 0
    index = 0
    
    while index < N:
        T += 1
        if T % 3 == 0:
            healths[index] -= 3
        else:
            healths[index] -= 1
        
        if healths[index] <= 0:
            index += 1
    
    return T

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
healths = list(map(int, data[1:N+1]))

result = calculate_time_to_defeat_enemies(N, healths)
print(result)