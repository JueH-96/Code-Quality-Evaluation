def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    
    closure = [[False] * n for _ in range(n)]
    for i in range(n):
        closure[i][i] = True
        
    index = 2
    for _ in range(m):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        closure[a-1][b-1] = True
        
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if closure[i][k] and closure[k][j]:
                    closure[i][j] = True
                    
    candidates = []
    for j in range(n):
        has_incoming = False
        for i in range(n):
            if i != j and closure[i][j]:
                has_incoming = True
                break
        if not has_incoming:
            candidates.append(j)
            
    if len(candidates) == 1:
        print(candidates[0] + 1)
    else:
        print(-1)

if __name__ == "__main__":
    main()