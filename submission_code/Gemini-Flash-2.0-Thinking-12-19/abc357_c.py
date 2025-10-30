def solve():
    n = int(input())
    if n == 0:
        print("#")
        return
    
    size = 3**n
    
    def get_char(level, row, col):
        if level == 0:
            return '#'
        size_prev = 3**(level - 1)
        block_row_index = (row + size_prev - 1) // size_prev
        block_col_index = (col + size_prev - 1) // size_prev
        if block_row_index == 2 and block_col_index == 2:
            return '.'
        else:
            row_in_block = row - (block_row_index - 1) * size_prev
            col_in_block = col - (block_col_index - 1) * size_prev
            return get_char(level - 1, row_in_block, col_in_block)
            
    for i in range(1, size + 1):
        row_str = ""
        for j in range(1, size + 1):
            char = get_char(n, i, j)
            row_str += char
        print(row_str)

if __name__ == '__main__':
    solve()