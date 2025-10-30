# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    adjacency_matrix = []
    index = 1
    for i in range(N):
        row = list(map(int, data[index:index+N]))
        adjacency_matrix.append(row)
        index += N
    
    for i in range(N):
        connected = []
        for j in range(N):
            if adjacency_matrix[i][j] == 1:
                connected.append(str(j+1))
        print(' '.join(connected))

if __name__ == "__main__":
    main()