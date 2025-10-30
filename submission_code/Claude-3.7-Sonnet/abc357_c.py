def is_black(i, j, k):
    if k == 0:
        return True  # Level-0 carpet is a single black cell
    
    sub_size = 3 ** (k - 1)
    
    # Check if the point is in the central block
    if i // sub_size == 1 and j // sub_size == 1:
        return False  # Central block is white
    
    # Recursively check the sub-block
    return is_black(i % sub_size, j % sub_size, k - 1)

def print_carpet(n):
    size = 3 ** n
    for i in range(size):
        row = ""
        for j in range(size):
            if is_black(i, j, n):
                row += "#"
            else:
                row += "."
        print(row)

n = int(input())
print_carpet(n)