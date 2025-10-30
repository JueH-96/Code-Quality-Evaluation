import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    matrix = [list(map(int, data[i * n + 1:(i + 1) * n + 1])) for i in range(n)]
    
    for i in range(n):
        connections = [str(j + 1) for j in range(n) if matrix[i][j] == 1]
        print(" ".join(connections))

if __name__ == "__main__":
    main()