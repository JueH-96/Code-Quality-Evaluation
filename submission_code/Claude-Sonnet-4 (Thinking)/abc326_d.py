n = int(input())
R = input().strip()
C = input().strip()

# Initialize grid
grid = [['' for _ in range(n)] for _ in range(n)]

# Keep track of characters placed in each row and column
row_chars = [set() for _ in range(n)]
col_chars = [set() for _ in range(n)]

def get_leftmost_char(i):
    for j in range(n):
        if grid[i][j] != '':
            return grid[i][j]
    return None

def get_topmost_char(j):
    for i in range(n):
        if grid[i][j] != '':
            return grid[i][j]
    return None

def is_valid():
    for i in range(n):
        if len(row_chars[i]) > 0:
            if get_leftmost_char(i) != R[i]:
                return False
    for j in range(n):
        if len(col_chars[j]) > 0:
            if get_topmost_char(j) != C[j]:
                return False
    return True

def is_complete():
    for i in range(n):
        if len(row_chars[i]) != 3:
            return False
    for j in range(n):
        if len(col_chars[j]) != 3:
            return False
    return True

def backtrack():
    if is_complete():
        return True
    
    for i in range(n):
        for j in range(n):
            if grid[i][j] == '':
                for char in ['A', 'B', 'C']:
                    if char not in row_chars[i] and char not in col_chars[j]:
                        grid[i][j] = char
                        row_chars[i].add(char)
                        col_chars[j].add(char)
                        
                        if is_valid():
                            if backtrack():
                                return True
                        
                        grid[i][j] = ''
                        row_chars[i].remove(char)
                        col_chars[j].remove(char)
                
                return False
    
    return False

if backtrack():
    print("Yes")
    for i in range(n):
        row_str = ""
        for j in range(n):
            if grid[i][j] == '':
                row_str += '.'
            else:
                row_str += grid[i][j]
        print(row_str)
else:
    print("No")