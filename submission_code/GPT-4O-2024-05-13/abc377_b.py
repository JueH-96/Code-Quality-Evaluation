# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    rows_with_pieces = set()
    cols_with_pieces = set()
    
    for i in range(8):
        for j in range(8):
            if data[i][j] == '#':
                rows_with_pieces.add(i)
                cols_with_pieces.add(j)
    
    safe_squares = 0
    for i in range(8):
        for j in range(8):
            if data[i][j] == '.' and i not in rows_with_pieces and j not in cols_with_pieces:
                safe_squares += 1
    
    print(safe_squares)

if __name__ == "__main__":
    main()