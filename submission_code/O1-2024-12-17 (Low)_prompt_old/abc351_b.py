def solve():
    import sys
    data = sys.stdin.read().strip().split()
    
    # First integer is N
    N = int(data[0])
    
    # Next N lines are grid A
    A = data[1:1+N]
    # Next N lines are grid B
    B = data[1+N:1+2*N]
    
    # Find the differing cell
    for i in range(N):
        for j in range(N):
            if A[i][j] != B[i][j]:
                print(i+1, j+1)
                return

def main():
    solve()
    
if __name__ == "__main__":
    main()