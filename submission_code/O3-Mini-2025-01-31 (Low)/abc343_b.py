def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    index = 0
    N = int(input_data[index])
    index += 1
    matrix = []
    for _ in range(N):
        row = []
        for _ in range(N):
            row.append(int(input_data[index]))
            index += 1
        matrix.append(row)
    
    output_lines = []
    for i in range(N):
        # vertices connected to vertex (i+1)
        connected = []
        for j in range(N):
            if matrix[i][j] == 1:
                connected.append(str(j+1))
        output_lines.append(" ".join(connected))
    
    sys.stdout.write("
".join(output_lines))
    
if __name__ == "__main__":
    main()