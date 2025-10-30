import itertools

def solve():
    n = int(input())
    r_str = input()
    c_str = input()
    chars = ['A', 'B', 'C']
    
    permutations_n = list(itertools.permutations(range(n)))
    
    for perm_indices in itertools.product(permutations_n, permutations_n, permutations_n):
        pi_a_indices, pi_b_indices, pi_c_indices = perm_indices
        valid_perms = True
        for i in range(n):
            if not (pi_a_indices[i] != pi_b_indices[i] and pi_a_indices[i] != pi_c_indices[i] and pi_b_indices[i] != pi_c_indices[i]):
                valid_perms = False
                break
        if not valid_perms:
            continue
            
        grid = [['.' for _ in range(n)] for _ in range(n)]
        for i in range(n):
            grid[i][pi_a_indices[i]] = 'A'
            grid[i][pi_b_indices[i]] = 'B'
            grid[i][pi_c_indices[i]] = 'C'
            
        row_first_chars = []
        for i in range(n):
            first_char = None
            for j in range(n):
                if grid[i][j] != '.':
                    first_char = grid[i][j]
                    break
            if first_char is None:
                first_char = '.' # Should not happen if permutations are valid
            row_first_chars.append(first_char)
            
        col_first_chars = []
        for j in range(n):
            first_char = None
            for i in range(n):
                if grid[i][j] != '.':
                    first_char = grid[i][j]
                    break
            if first_char is None:
                first_char = '.' # Should not happen if permutations are valid
            col_first_chars.append(first_char)
            
        row_condition_met = True
        for i in range(n):
            if row_first_chars[i] != r_str[i]:
                row_condition_met = False
                break
                
        col_condition_met = True
        for j in range(n):
            if col_first_chars[j] != c_str[j]:
                col_condition_met = False
                break
                
        if row_condition_met and col_condition_met:
            print("Yes")
            for i in range(n):
                print("".join(grid[i]))
            return
            
    print("No")

if __name__ == '__main__':
    solve()