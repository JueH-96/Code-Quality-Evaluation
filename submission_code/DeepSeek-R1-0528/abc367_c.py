def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    R = list(map(int, data[2:2+n]))
    path = []
    
    def dfs(i, cur):
        if i == n:
            if cur % k == 0:
                print(" ".join(map(str, path)))
            return
        for x in range(1, R[i] + 1):
            path.append(x)
            new_cur = (cur + x) % k
            dfs(i+1, new_cur)
            path.pop()
    
    dfs(0, 0)

if __name__ == "__main__":
    main()