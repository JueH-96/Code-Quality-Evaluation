# YOUR CODE HERE
def generate_carpet(n):
    if n == 0:
        return ['#']
    
    prev_carpet = generate_carpet(n-1)
    size = 3 ** n
    
    new_carpet = []
    for i in range(size):
        row = ''
        for j in range(size):
            block_size = 3 ** (n-1)
            if (i // block_size) == 1 and (j // block_size) == 1:
                row += '.'
            else:
                row += prev_carpet[(i % block_size)][(j % block_size)]
        new_carpet.append(row)
    
    return new_carpet

n = int(input())
carpet = generate_carpet(n)
for row in carpet:
    print(row)