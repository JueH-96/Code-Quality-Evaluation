# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    black_rows = {}
    white_rows = {}
    black_cols = {}
    white_cols = {}
    
    index = 2
    for _ in range(M):
        X = int(data[index])
        Y = int(data[index + 1])
        C = data[index + 2]
        index += 3
        
        if C == 'B':
            if X not in black_rows:
                black_rows[X] = Y
            else:
                black_rows[X] = max(black_rows[X], Y)
            
            if Y not in black_cols:
                black_cols[Y] = X
            else:
                black_cols[Y] = max(black_cols[Y], X)
        else:
            if X not in white_rows:
                white_rows[X] = Y
            else:
                white_rows[X] = min(white_rows[X], Y)
            
            if Y not in white_cols:
                white_cols[Y] = X
            else:
                white_cols[Y] = min(white_cols[Y], X)
    
    for row in black_rows:
        if row in white_rows and black_rows[row] >= white_rows[row]:
            print("No")
            return
    
    for col in black_cols:
        if col in white_cols and black_cols[col] >= white_cols[col]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()