def is_black(N, i, j):
    if N == 0:
        return True
    
    block_size = 3 ** (N - 1)
    block_row = i // block_size
    block_col = j // block_size
    
    if block_row == 1 and block_col == 1:
        # Central block: white
        return False
    else:
        # Recursively determine the color in the level-(N-1) carpet
        return is_black(N - 1, i % block_size, j % block_size)

N = int(input())
size = 3 ** N
for i in range(size):
    row = ""
    for j in range(size):
        if is_black(N, i, j):
            row += "#"
        else:
            row += "."
    print(row)