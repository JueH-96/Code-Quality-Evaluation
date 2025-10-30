import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N, M = int(data[0]), int(data[1])
    A = list(map(int, data[2:M+2]))
    X = [list(map(int, data[i*M+M+2:(i+1)*M+M+2])) for i in range(N)]
    
    total_nutrients = [0] * M
    for i in range(N):
        for j in range(M):
            total_nutrients[j] += X[i][j]
    
    for i in range(M):
        if total_nutrients[i] < A[i]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()