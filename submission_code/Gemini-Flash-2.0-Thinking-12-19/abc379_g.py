def solve():
    h, w = map(int, input().split())
    s_grid = [list(input()) for _ in range(h)]
    
    def is_valid_grid(grid):
        for i in range(h):
            for j in range(w):
                digit = int(grid[i][j])
                if i > 0 and int(grid[i-1][j]) == digit:
                    return False
                if j > 0 and int(grid[i][j-1]) == digit:
                    return False
        return True
        
    question_marks = []
    for i in range(h):
        for j in range(w):
            if s_grid[i][j] == '?':
                question_marks.append((i, j))
                
    count = 0
    num_question_marks = len(question_marks)
    
    import itertools
    
    possible_replacements = []
    for _ in range(num_question_marks):
        possible_replacements.append(['1', '2', '3'])
        
    for replacements in itertools.product(*possible_replacements):
        current_grid = [row[:] for row in s_grid]
        for index, pos in enumerate(question_marks):
            r, c = pos
            current_grid[r][c] = replacements[index]
            
        if is_valid_grid(current_grid):
            count += 1
            
    print(count % 998244353)

if __name__ == '__main__':
    solve()