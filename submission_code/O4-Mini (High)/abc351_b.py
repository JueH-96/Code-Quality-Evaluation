def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = [input().rstrip() for _ in range(N)]
    B = [input().rstrip() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i][j] != B[i][j]:
                # Found the differing cell; print 1-based indices
                print(i + 1, j + 1)
                return

# Call main to execute    
main()