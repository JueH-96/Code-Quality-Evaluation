import sys

def main():
    lines = sys.stdin.read().splitlines()
    N = int(lines[0])
    adjacency_matrix = [list(map(int, line.split())) for line in lines[1:N+1]]
    
    for i in range(1, N+1):
        connections = [str(j+1) for j in range(N) if adjacency_matrix[i-1][j] == 1]
        print(' '.join(connections))

if __name__ == "__main__":
    main()