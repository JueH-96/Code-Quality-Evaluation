def solve():
    n = int(input())
    center_row = (n + 1) // 2
    center_col = (n + 1) // 2
    grid = [[0] * n for _ in range(n)]
    grid[center_row-1][center_col-1] = 'T'
    
    current_number = 1
    current_row, current_col = 0, 0
    
    for layer in range((n + 1) // 2):
        length = n - 2 * layer - 1
        if length <= 0:
            break
            
        # Right
        for _ in range(length):
            current_col += 1
            if current_row + 1 != center_row or current_col + 1 != center_col:
                grid[current_row][current_col] = current_number
                current_number += 1
                
        # Down
        for _ in range(length):
            current_row += 1
            if current_row + 1 != center_row or current_col + 1 != center_col:
                grid[current_row][current_col] = current_number
                current_number += 1
                
        # Left
        for _ in range(length):
            current_col -= 1
            if current_row + 1 != center_row or current_col + 1 != center_col:
                grid[current_row][current_col] = current_number
                current_number += 1
                
        # Up
        for _ in range(length):
            current_row -= 1
            if current_row + 1 != center_row or current_col + 1 != center_col:
                grid[current_row][current_col] = current_number
                current_number += 1
                
    for i in range(n):
        print(*(grid[i]))

if __name__ == '__main__':
    solve()