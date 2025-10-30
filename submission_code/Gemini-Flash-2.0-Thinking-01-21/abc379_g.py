def solve():
    h, w = map(int, input().split())
    s_grid = [list(input()) for _ in range(h)]
    
    def is_valid_placement(r, c, digit, current_grid):
        if r > 0 and current_grid[r-1][c] != '?' and int(current_grid[r-1][c]) == digit:
            return False
        if c > 0 and current_grid[r][c-1] != '?' and int(current_grid[r][c-1]) == digit:
            return False
        return True
        
    def get_valid_digits(r, c, current_grid):
        valid_digits = []
        for digit in [1, 2, 3]:
            if is_valid_placement(r, c, digit, current_grid):
                valid_digits.append(digit)
        return valid_digits
        
    memo = {}
    
    def count_valid_completions(r, c, current_grid):
        if r == h:
            return 1
        if c == w:
            return count_valid_completions(r + 1, 0, current_grid)
            
        if s_grid[r][c] != '?':
            digit = int(s_grid[r][c])
            if not is_valid_placement(r, c, digit, current_grid):
                return 0
            current_grid[r][c] = s_grid[r][c]
            result = count_valid_completions(r, c + 1, current_grid)
            current_grid[r][c] = '?' # backtrack not needed as original grid char was not '?'
            return result
        else:
            count = 0
            for digit in get_valid_digits(r, c, current_grid):
                current_grid[r][c] = str(digit)
                count = (count + count_valid_completions(r, c + 1, current_grid)) % 998244353
                current_grid[r][c] = '?' # backtrack
            return count

    initial_grid = [['?'] * w for _ in range(h)]
    result = count_valid_completions(0, 0, initial_grid)
    print(result)

if __name__ == '__main__':
    solve()