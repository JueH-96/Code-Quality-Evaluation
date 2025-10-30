import sys

def solve():
    n = int(sys.stdin.readline())
    r_str = sys.stdin.readline().strip()
    c_str = sys.stdin.readline().strip()
    r = list(r_str)
    c = list(c_str)
    chars = ['A', 'B', 'C']
    import itertools
    
    permutations = list(itertools.permutations(range(1, n + 1)))
    
    for sigma_a_tuple in permutations:
        sigma_a = list(sigma_a_tuple)
        for sigma_b_tuple in permutations:
            sigma_b = list(sigma_b_tuple)
            for sigma_c_tuple in permutations:
                sigma_c = list(sigma_c_tuple)
                
                valid_config = True
                inverse_sigma_a = [0] * (n + 1)
                inverse_sigma_b = [0] * (n + 1)
                inverse_sigma_c = [0] * (n + 1)
                for i in range(n):
                    inverse_sigma_a[sigma_a[i]] = i + 1
                    inverse_sigma_b[sigma_b[i]] = i + 1
                    inverse_sigma_c[sigma_c[i]] = i + 1
                    
                for j in range(1, n + 1):
                    rows = {inverse_sigma_a[j], inverse_sigma_b[j], inverse_sigma_c[j]}
                    if len(rows) != 3:
                        valid_config = False
                        break
                if not valid_config:
                    continue
                    
                first_col_condition_met = True
                for j in range(1, n + 1):
                    row_indices = [inverse_sigma_a[j], inverse_sigma_b[j], inverse_sigma_c[j]]
                    min_row_index = min(row_indices)
                    char_at_min_row = ''
                    if min_row_index == inverse_sigma_a[j]:
                        char_at_min_row = 'A'
                    elif min_row_index == inverse_sigma_b[j]:
                        char_at_min_row = 'B'
                    else:
                        char_at_min_row = 'C'
                    if char_at_min_row != c[j-1]:
                        first_col_condition_met = False
                        break
                if not first_col_condition_met:
                    continue
                    
                first_row_condition_met = True
                for i in range(n):
                    col_indices = [sigma_a[i], sigma_b[i], sigma_c[i]]
                    min_col_index = min(col_indices)
                    char_at_min_col = ''
                    if min_col_index == sigma_a[i]:
                        char_at_min_col = 'A'
                    elif min_col_index == sigma_b[i]:
                        char_at_min_col = 'B'
                    else:
                        char_at_min_col = 'C'
                    if char_at_min_col != r[i]:
                        first_row_condition_met = False
                        break
                if not first_row_condition_met:
                    continue
                    
                if valid_config and first_col_condition_met and first_row_condition_met:
                    print("Yes")
                    grid = [['.' for _ in range(n)] for _ in range(n)]
                    for i in range(n):
                        grid[i][sigma_a[i]-1] = 'A'
                        grid[i][sigma_b[i]-1] = 'B'
                        grid[i][sigma_c[i]-1] = 'C'
                    for i in range(n):
                        print("".join(grid[i]))
                    return
                    
    print("No")

if __name__ == '__main__':
    solve()