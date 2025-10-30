# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

head_x = 1
head_y = 0
parts = [(i, 0) for i in range(1, N + 1)]

index = 2
results = []

for _ in range(Q):
    query = data[index]
    index += 1
    
    if query == '1':
        direction = data[index]
        index += 1
        
        if direction == 'R':
            head_x += 1
        elif direction == 'L':
            head_x -= 1
        elif direction == 'U':
            head_y += 1
        elif direction == 'D':
            head_y -= 1
        
        for i in range(1, N):
            parts[i] = parts[i - 1]
    
    elif query == '2':
        p = int(data[index])
        index += 1
        results.append(parts[p - 1])

for result in results:
    print(result[0], result[1])