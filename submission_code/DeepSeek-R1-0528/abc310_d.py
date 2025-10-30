import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    T = int(data[1])
    m = int(data[2])
    bad_mat = [[False] * n for _ in range(n)]
    
    index = 3
    for _ in range(m):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        a0 = a - 1
        b0 = b - 1
        bad_mat[a0][b0] = True
        bad_mat[b0][a0] = True
        
    groups = []
    
    def dfs(i):
        if i == n:
            if len(groups) == T:
                return 1
            return 0
            
        k = len(groups)
        if n - i < T - k:
            return 0
            
        total = 0
        for idx in range(k):
            conflict = False
            for j in groups[idx]:
                if bad_mat[i][j]:
                    conflict = True
                    break
            if not conflict:
                groups[idx].append(i)
                total += dfs(i + 1)
                groups[idx].pop()
                
        if k < T:
            groups.append([i])
            total += dfs(i + 1)
            groups.pop()
            
        return total
        
    result = dfs(0)
    print(result)

if __name__ == "__main__":
    main()