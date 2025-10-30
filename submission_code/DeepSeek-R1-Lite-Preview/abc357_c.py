def is_white(row, col, N):
    size = 3**(N-1)
    while N > 0:
        row_div = row // size
        col_div = col // size
        if row_div == 1 and col_div == 1:
            return True
        row = row % size
        col = col % size
        N -= 1
        size = size // 3
    return False

def main():
    N = int(input().strip())
    size = 3**N
    for row in range(size):
        line = []
        for col in range(size):
            if is_white(row, col, N):
                line.append('.')
            else:
                line.append('#')
        print(''.join(line))

if __name__ == "__main__":
    main()