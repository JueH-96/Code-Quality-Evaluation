def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    size = 3 ** N
    
    # We determine the value for each cell by checking if, at any step,
    # the cell lies in the central block (position where both indices modulo 3 equal 1).
    # If it does, then the cell is white ('.'), otherwise black ('#').
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            x, y = i, j
            char = '#'
            while x > 0 or y > 0:
                if x % 3 == 1 and y % 3 == 1:
                    char = '.'
                    break
                x //= 3
                y //= 3
            row.append(char)
        result.append(''.join(row))
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()