def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    mat = []
    index = 1
    for i in range(1, n + 1):
        row = list(map(int, data[index].split()))
        index += 1
        mat.append(row)
    
    current = 1
    for j in range(1, n + 1):
        a = current
        b = j
        if a >= b:
            current = mat[a - 1][b - 1]
        else:
            current = mat[b - 1][a - 1]
    
    print(current)

if __name__ == "__main__":
    main()