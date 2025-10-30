import math

def solve():
    h, w, k = map(int, input().split())
    s = [input() for _ in range(h)]
    min_operations = float('inf')
    found_solution = False
    
    # Check horizontal sequences
    for i in range(h):
        for j in range(w - k + 1):
            is_valid_sequence = True
            operations_needed = 0
            for col in range(j, j + k):
                if s[i][col] == 'x':
                    is_valid_sequence = False
                    break
                elif s[i][col] == '.':
                    operations_needed += 1
            if is_valid_sequence:
                found_solution = True
                min_operations = min(min_operations, operations_needed)
                
    # Check vertical sequences
    for j in range(w):
        for i in range(h - k + 1):
            is_valid_sequence = True
            operations_needed = 0
            for row in range(i, i + k):
                if s[row][j] == 'x':
                    is_valid_sequence = False
                    break
                elif s[row][j] == '.':
                    operations_needed += 1
            if is_valid_sequence:
                found_solution = True
                min_operations = min(min_operations, operations_needed)
                
    if not found_solution:
        print("-1")
    else:
        print(min_operations)

if __name__ == '__main__':
    solve()