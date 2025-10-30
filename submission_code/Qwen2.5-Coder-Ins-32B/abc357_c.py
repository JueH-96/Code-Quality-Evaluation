# YOUR CODE HERE
def generate_carpet(N):
    if N == 0:
        return ["#"]
    
    prev_carpet = generate_carpet(N - 1)
    size = len(prev_carpet)
    new_size = size * 3
    
    new_carpet = []
    
    for i in range(new_size):
        row = []
        for j in range(new_size):
            if size <= i < size * 2 and size <= j < size * 2:
                row.append('.')
            else:
                row.append(prev_carpet[i // 3][j // 3])
        new_carpet.append(''.join(row))
    
    return new_carpet

import sys
input = sys.stdin.read().strip()
N = int(input)
carpet = generate_carpet(N)
for line in carpet:
    print(line)