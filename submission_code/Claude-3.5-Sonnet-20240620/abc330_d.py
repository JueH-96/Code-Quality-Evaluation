# YOUR CODE HERE
def count_triples():
    N = int(input())
    grid = [input().strip() for _ in range(N)]
    
    count = 0
    
    # Helper function to check if a cell contains 'o'
    def is_o(i, j):
        return 0 <= i < N and 0 <= j < N and grid[i][j] == 'o'
    
    # Check all possible triples
    for i in range(N):
        for j in range(N):
            if is_o(i, j):
                # Case 1: Two in the same row, one in a different row
                for k in range(j+1, N):
                    if is_o(i, k):
                        for l in range(N):
                            if l != i and is_o(l, j):
                                count += 1
                            if l != i and is_o(l, k):
                                count += 1
                
                # Case 2: Two in the same column, one in a different column
                for k in range(i+1, N):
                    if is_o(k, j):
                        for l in range(N):
                            if l != j and is_o(i, l):
                                count += 1
                            if l != j and is_o(k, l):
                                count += 1
    
    print(count)

count_triples()