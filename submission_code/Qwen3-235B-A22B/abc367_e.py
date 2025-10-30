def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    k = int(data[idx])
    idx += 1
    
    x = list(map(int, data[idx:idx+n]))
    idx += n
    a = list(map(int, data[idx:idx+n]))
    
    f = [xi - 1 for xi in x]
    max_level = 60
    jump = [[0] * n for _ in range(max_level)]
    
    for i in range(n):
        jump[0][i] = f[i]
    
    for level in range(1, max_level):
        for i in range(n):
            jump[level][i] = jump[level-1][jump[level-1][i]]
    
    res = []
    for i in range(n):
        current = i
        for b in range(max_level):
            if k & (1 << b):
                current = jump[b][current]
        res.append(str(a[current]))
    
    print(' '.join(res))

if __name__ == "__main__":
    main()