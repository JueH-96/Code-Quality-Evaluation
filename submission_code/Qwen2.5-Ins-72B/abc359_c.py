# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    Sx = int(data[0])
    Sy = int(data[1])
    Tx = int(data[2])
    Ty = int(data[3])
    
    def get_tile(x, y):
        if (x + y) % 2 == 0:
            return (x // 2, y)
        else:
            return (x // 2, y // 2)
    
    start_tile = get_tile(Sx, Sy)
    end_tile = get_tile(Tx, Ty)
    
    toll = abs(start_tile[0] - end_tile[0]) + abs(start_tile[1] - end_tile[1])
    print(toll)

if __name__ == "__main__":
    main()