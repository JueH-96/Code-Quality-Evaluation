def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    original = input[1:N+1]
    
    result = [[0 for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            a = i + 1
            b = j + 1
            layer = min(a, b, N + 1 - a, N + 1 - b)
            m = layer
            mod = m % 4
            if mod == 1:
                x1 = N + 1 - b
                y1 = a
            elif mod == 2:
                x1 = N + 1 - a
                y1 = N + 1 - b
            elif mod == 3:
                x1 = b
                y1 = N + 1 - a
            else:
                x1 = a
                y1 = b
            x = x1 - 1
            y = y1 - 1
            result[i][j] = original[x][y]
    
    for row in result:
        print(''.join(row))

if __name__ == "__main__":
    main()