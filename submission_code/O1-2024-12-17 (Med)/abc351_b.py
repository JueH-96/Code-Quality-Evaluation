def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = data[1:N+1]
    B = data[N+1:2*N+1]
    
    # Find the single (i, j) where A[i][j] != B[i][j]
    for i in range(N):
        for j in range(N):
            if A[i][j] != B[i][j]:
                print(i+1, j+1)
                return

# Do not forget to call main.
main()