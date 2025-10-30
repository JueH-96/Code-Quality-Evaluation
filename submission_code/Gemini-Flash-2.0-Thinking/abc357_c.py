def solve():
    n = int(input())
    size = 3**n

    def is_black(k, r, c):
        if k == 0:
            return True
        sub_size = 3**(k - 1)
        row_block = r // sub_size
        col_block = c // sub_size
        if row_block == 1 and col_block == 1:
            return False
        else:
            return is_black(k - 1, r % sub_size, c % sub_size)

    for r in range(size):
        row_str = ""
        for c in range(size):
            if is_black(n, r, c):
                row_str += "#"
            else:
                row_str += "."
        print(row_str)

solve()