def remaining_balls(N, A):
    stack = []
    
    for i in range(N):
        size = 2 ** A[i]
        stack.append(size)
        
        while len(stack) > 1 and stack[-1] == stack[-2]:
            # Remove the last two balls and add a new one
            last_size = stack.pop()
            stack.pop()  # remove the second last
            new_size = last_size + last_size  # sum of the two sizes
            stack.append(new_size)  # add the new ball
            
    return len(stack)

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

print(remaining_balls(N, A))