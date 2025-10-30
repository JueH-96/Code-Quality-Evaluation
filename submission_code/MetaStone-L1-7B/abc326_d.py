def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    R = input[idx]
    idx += 1
    C = input[idx]
    
    if R[0] != C[0]:
        print("No")
        return
    
    grid = [['.' for _ in range(N)] for _ in range(N)]
    column_counts = [[0, 0, 0] for _ in range(N)]
    
    def backtrack(row):
        if row == N:
            return grid
        current_char = R[row]
        other_letters = []
        if current_char == 'A':
            other_letters = ['B', 'C']
        elif current_char == 'B':
            other_letters = ['A', 'C']
        else:
            other_letters = ['A', 'B']
        
        for j1 in range(1, N):
            for j2 in range(j1 + 1, N):
                # Try assigning other_letters[0] to j1, other_letters[1] to j2
                valid = True
                if column_counts[j1][0] > 0:
                    valid = False
                if column_counts[j2][1] > 0:
                    valid = False
                if valid:
                    new_grid = [row.copy() for row in grid]
                    new_grid[row][j1] = other_letters[0]
                    new_grid[row][j2] = other_letters[1]
                    new_counts = [col.copy() for col in column_counts]
                    new_counts[j1][0] += 1
                    new_counts[j2][1] += 1
                    result = backtrack(row + 1)
                    if result is not None:
                        return result
                
                # Try assigning other_letters[1] to j1, other_letters[0] to j2
                valid = True
                if column_counts[j1][1] > 0:
                    valid = False
                if column_counts[j2][0] > 0:
                    valid = False
                if valid:
                    new_grid = [row.copy() for row in grid]
                    new_grid[row][j1] = other_letters[1]
                    new_grid[row][j2] = other_letters[0]
                    new_counts = [col.copy() for col in column_counts]
                    new_counts[j1][1] += 1
                    new_counts[j2][0] += 1
                    result = backtrack(row + 1)
                    if result is not None:
                        return result
        
        return None
    
    solution = backtrack(0)
    if solution is not None:
        for row in solution:
            print(''.join(row))
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()