import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    T = int(data[1])
    m = int(data[2])
    
    bad = [[False] * n for _ in range(n)]
    idx = 3
    for _ in range(m):
        a = int(data[idx])
        b = int(data[idx+1])
        idx += 2
        a -= 1
        b -= 1
        bad[a][b] = True
        bad[b][a] = True
        
    group_assignment = [-1] * n
    
    def dfs(i, groups_count):
        remaining = n - i
        if groups_count + remaining < T:
            return 0
        if i == n:
            return 1 if groups_count == T else 0
            
        total = 0
        for g in range(groups_count):
            valid = True
            for j in range(i):
                if group_assignment[j] == g and bad[i][j]:
                    valid = False
                    break
            if valid:
                group_assignment[i] = g
                total += dfs(i+1, groups_count)
                group_assignment[i] = -1
                
        if groups_count < T:
            group_assignment[i] = groups_count
            total += dfs(i+1, groups_count + 1)
            group_assignment[i] = -1
            
        return total
        
    ans = dfs(0, 0)
    print(ans)

if __name__ == "__main__":
    main()