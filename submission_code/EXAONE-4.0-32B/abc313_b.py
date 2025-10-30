import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    
    closure = [[False] * (n+1) for _ in range(n+1)]
    
    index = 2
    for _ in range(m):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        closure[a][b] = True
        
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if closure[i][k] and closure[k][j]:
                    closure[i][j] = True
                    
    candidates = []
    for x in range(1, n+1):
        valid = True
        for y in range(1, n+1):
            if x == y:
                continue
            if not closure[x][y]:
                valid = False
                break
        if valid:
            candidates.append(x)
            
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()