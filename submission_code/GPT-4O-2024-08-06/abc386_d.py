# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    max_black_row = {}
    min_white_row = {}
    max_black_col = {}
    min_white_col = {}
    
    index = 2
    for _ in range(M):
        X = int(data[index]) - 1
        Y = int(data[index + 1]) - 1
        C = data[index + 2]
        index += 3
        
        if C == 'B':
            if X not in max_black_row:
                max_black_row[X] = Y
            else:
                max_black_row[X] = max(max_black_row[X], Y)
            
            if Y not in max_black_col:
                max_black_col[Y] = X
            else:
                max_black_col[Y] = max(max_black_col[Y], X)
        
        elif C == 'W':
            if X not in min_white_row:
                min_white_row[X] = Y
            else:
                min_white_row[X] = min(min_white_row[X], Y)
            
            if Y not in min_white_col:
                min_white_col[Y] = X
            else:
                min_white_col[Y] = min(min_white_col[Y], X)
    
    # Check row constraints
    for r in range(N):
        if r in max_black_row and r in min_white_row:
            if max_black_row[r] >= min_white_row[r]:
                print("No")
                return
    
    # Check column constraints
    for c in range(N):
        if c in max_black_col and c in min_white_col:
            if max_black_col[c] >= min_white_col[c]:
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()