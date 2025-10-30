def main():
    import sys
    data = sys.stdin.read().strip().split()
    A = int(data[0])
    B = int(data[1])
    
    # Check if B is exactly A+1 and A is not the last element in its row (i.e., not at the right edge of the board)
    # The numbers are arranged as:
    # 1 2 3
    # 4 5 6
    # 7 8 9
    # So if A % 3 == 0, A is at the right edge.
    if B == A + 1 and A % 3 != 0:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()