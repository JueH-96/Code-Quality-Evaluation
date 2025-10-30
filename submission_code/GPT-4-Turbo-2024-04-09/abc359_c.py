def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    Sx = int(data[0])
    Sy = int(data[1])
    Tx = int(data[2])
    Ty = int(data[3])
    
    # Calculate the number of tiles crossed horizontally and vertically
    horizontal_crosses = abs(Tx - Sx)
    vertical_crosses = abs(Ty - Sy)
    
    # The minimum toll is the sum of the number of tiles crossed in each direction
    print(horizontal_crosses + vertical_crosses)

if __name__ == "__main__":
    main()