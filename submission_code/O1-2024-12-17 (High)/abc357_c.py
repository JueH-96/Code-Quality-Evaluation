# YOUR CODE HERE
def main():
    import sys
    N = int(sys.stdin.readline().strip())
    size = 3 ** N

    for i in range(size):
        row_chars = []
        for j in range(size):
            x, y = i, j
            cell = '#'
            for _ in range(N):
                if x % 3 == 1 and y % 3 == 1:
                    cell = '.'
                    break
                x //= 3
                y //= 3
            row_chars.append(cell)
        print("".join(row_chars))

main()