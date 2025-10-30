import sys

def main():
    lines = sys.stdin.read().splitlines()
    N = int(lines[0])
    A = lines[1:1+N]
    B = lines[1+N:1+2*N]
    
    for i in range(N):
        for j in range(N):
            if A[i][j] != B[i][j]:
                print(i+1, j+1)
                return

if __name__ == "__main__":
    main()