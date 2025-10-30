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
    col_first_set = [False] * N
    col_letters = [set() for _ in range(N)]
    
    def dfs(row):
        if row == N:
            return True
            
        remaining_letters = list(set(['A','B','C']) - {R[row]})
        for cols in itertools.combinations(range(N), 3):
            for perm in itertools.permutations(remaining_letters):
                assign = {
                    cols[0]: R[row],
                    cols[1]: perm[0],
                    cols[2]: perm[1]
                }
                valid = True
                for j in cols:
                    ch = assign[j]
                    if ch in col_letters[j]:
                        valid = False
                        break
                    if not col_first_set[j]:
                        if ch != C[j]:
                            valid = False
                            break
                if not valid:
                    continue
                
                changed_first_set = []
                for j in cols:
                    grid[row][j] = assign[j]
                    col_letters[j].add(assign[j])
                    if not col_first_set[j]:
                        col_first_set[j] = True
                        changed_first_set.append(j)
                
                if dfs(row + 1):
                    return True
                    
                for j in cols:
                    grid[row][j] = '.'
                    col_letters[j].remove(assign[j])
                for j in changed_first_set:
                    col_first_set[j] = False
                    
        return False
        
    if dfs(0):
        print("Yes")
        for row_arr in grid:
            print(''.join(row_arr))
    else:
        print("No")

if __name__ == '__main__':
    main()