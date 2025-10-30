import itertools
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    N = int(data[0].strip())
    R = data[1].strip()
    C = data[2].strip()
    
    grid = [['.'] * N for _ in range(N)]
    col_top = [None] * N
    col_set = [set() for _ in range(N)]
    
    def dfs(i):
        if i == N:
            return True
            
        for comb in itertools.combinations(range(N), 3):
            j0, j1, j2 = sorted(comb)
            remaining_letters = list(set(['A', 'B', 'C']) - {R[i]})
            for perm in itertools.permutations(remaining_letters):
                if R[i] in col_set[j0]:
                    continue
                if col_top[j0] is None and R[i] != C[j0]:
                    continue
                    
                if perm[0] in col_set[j1]:
                    continue
                if col_top[j1] is None and perm[0] != C[j1]:
                    continue
                    
                if perm[1] in col_set[j2]:
                    continue
                if col_top[j2] is None and perm[1] != C[j2]:
                    continue
                    
                saved_state = []
                letters = [R[i], perm[0], perm[1]]
                for idx, j in enumerate(comb):
                    saved_state.append((j, col_top[j], set(col_set[j])))
                    col_set[j].add(letters[idx])
                    if col_top[j] is None:
                        col_top[j] = letters[idx]
                
                grid[i][j0] = R[i]
                grid[i][j1] = perm[0]
                grid[i][j2] = perm[1]
                
                if dfs(i + 1):
                    return True
                
                for j, old_top, old_set in saved_state:
                    col_top[j] = old_top
                    col_set[j] = old_set
                grid[i][j0] = '.'
                grid[i][j1] = '.'
                grid[i][j2] = '.'
                
        return False
        
    if dfs(0):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")

if __name__ == "__main__":
    main()