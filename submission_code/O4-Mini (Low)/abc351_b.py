import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    A = [input().rstrip('
') for _ in range(N)]
    B = [input().rstrip('
') for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if A[i][j] != B[i][j]:
                # print 1-based indices
                print(i+1, j+1)
                return

if __name__ == "__main__":
    main()