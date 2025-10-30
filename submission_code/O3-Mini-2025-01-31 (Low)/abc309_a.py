def main():
    import sys
    input_data = sys.stdin.read().split()
    A = int(input_data[0])
    B = int(input_data[1])
    # Check if A and B are in the same row and consecutive horizontally.
    # On the board:
    # 1 2 3
    # 4 5 6
    # 7 8 9
    # A and B are adjacent horizontally if B == A + 1 and A is not at the right edge of a row (i.e. A % 3 != 0)
    if B == A + 1 and A % 3 != 0:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()