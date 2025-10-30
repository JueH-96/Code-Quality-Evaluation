def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    A, B = map(int, data.split())
    
    # The board is a 3x3 grid with numbers 1 to 9
    # 1 2 3
    # 4 5 6
    # 7 8 9
    
    # Horizontal adjacency on the same row
    # (1,2), (2,3), (4,5), (5,6), (7,8), (8,9)
    horizontal_pairs = {(1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9)}
    
    # Check if (A, B) is in the set of horizontal pairs
    if (A, B) in horizontal_pairs:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()