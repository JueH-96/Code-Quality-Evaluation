import collections

def solve():
    h, w = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))
    b = []
    for _ in range(h):
        b.append(list(map(int, input().split())))
    
    sorted_rows_a = []
    for i in range(h):
        sorted_rows_a.append(tuple(sorted(a[i])))
    sorted_rows_b = []
    for i in range(h):
        sorted_rows_b.append(tuple(sorted(b[i])))
    
    rows_a_counts = collections.Counter(sorted_rows_a)
    rows_b_counts = collections.Counter(sorted_rows_b)
    if rows_a_counts != rows_b_counts:
        print("-1")
        return
        
    sorted_cols_a = []
    for j in range(w):
        col = [a[i][j] for i in range(h)]
        sorted_cols_a.append(tuple(sorted(col)))
    sorted_cols_b = []
    for j in range(w):
        col = [b[i][j] for i in range(h)]
        sorted_cols_b.append(tuple(sorted(col)))
        
    cols_a_counts = collections.Counter(sorted_cols_a)
    cols_b_counts = collections.Counter(sorted_cols_b)
    if cols_a_counts != cols_b_counts:
        print("-1")
        return
        
    possible_row_matches = []
    for i in range(h):
        matches = []
        for j in range(h):
            if tuple(sorted(a[j])) == tuple(sorted(b[i])):
                matches.append(j)
        possible_row_matches.append(matches)
        
    possible_col_matches = []
    for j in range(w):
        matches = []
        for k in range(w):
            col_a = [a[i][k] for i in range(h)]
            col_b = [b[i][j] for i in range(h)]
            if tuple(sorted(col_a)) == tuple(sorted(col_b)):
                matches.append(k)
        possible_col_matches.append(matches)
        
    valid_row_permutations = []
    
    def get_row_permutations(index, current_perm, used_indices):
        if index == h:
            valid_row_permutations.append(tuple(current_perm))
            return
        for possible_index in possible_row_matches[index]:
            if possible_index not in used_indices:
                used_indices.add(possible_index)
                current_perm.append(possible_index)
                get_row_permutations(index + 1, current_perm, used_indices)
                current_perm.pop()
                used_indices.remove(possible_index)
                
    get_row_permutations(0, [], set())
    
    valid_col_permutations = []
    
    def get_col_permutations(index, current_perm, used_indices):
        if index == w:
            valid_col_permutations.append(tuple(current_perm))
            return
        for possible_index in possible_col_matches[index]:
            if possible_index not in used_indices:
                used_indices.add(possible_index)
                current_perm.append(possible_index)
                get_col_permutations(index + 1, current_perm, used_indices)
                current_perm.pop()
                used_indices.remove(possible_index)
                
    get_col_permutations(0, [], set())
    
    min_ops = float('inf')
    
    if not valid_row_permutations or not valid_col_permutations:
        print("-1")
        return
        
    for row_perm_indices in valid_row_permutations:
        row_perm = list(row_perm_indices)
        row_inversions = 0
        for i in range(h):
            for j in range(i + 1, h):
                if row_perm[i] > row_perm[j]:
                    row_inversions += 1
                    
        for col_perm_indices in valid_col_permutations:
            col_perm = list(col_perm_indices)
            col_inversions = 0
            for i in range(w):
                for j in range(i + 1, w):
                    if col_perm[i] > col_perm[j]:
                        col_inversions += 1
                        
            current_ops = row_inversions + col_inversions
            min_ops = min(min_ops, current_ops)
            
    if min_ops == float('inf'):
        print("-1")
    else:
        print(min_ops)

if __name__ == '__main__':
    solve()