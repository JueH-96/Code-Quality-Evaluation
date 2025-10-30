def generate_carpet(n):
    if n == 0:
        return ["#"]
    
    smaller_carpet = generate_carpet(n - 1)
    size = 3 ** (n - 1)
    new_size = 3 ** n
    carpet = []
    
    # Top 3 blocks (top-left, top-center, top-right)
    for i in range(size):
        carpet.append(smaller_carpet[i] * 3)
    
    # Middle 3 blocks (middle-left, middle-center (all white), middle-right)
    white_block = '.' * size
    for i in range(size):
        carpet.append(smaller_carpet[i] + white_block + smaller_carpet[i])
    
    # Bottom 3 blocks (bottom-left, bottom-center, bottom-right)
    for i in range(size):
        carpet.append(smaller_carpet[i] * 3)
    
    return carpet

import sys
input = sys.stdin.read
n = int(input().strip())
carpet = generate_carpet(n)
for line in carpet:
    print(line)