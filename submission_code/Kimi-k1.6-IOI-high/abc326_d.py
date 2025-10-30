import sys
import itertools

def generate_rows(R_char, N):
    rows = []
    other_chars = [c for c in ['A', 'B', 'C'] if c != R_char]
    for k in range(N):
        base_row = ['.' for _ in range(N)]
        base_row[k] = R_char
        available_positions = list(range(k+1, N))
        if len(available_positions) < 2:
            continue
        for m, n in itertools.combinations(available_positions, 2):
            row1 = base_row.copy()
            row1[m] = other_chars[0]
            row1[n] = other_chars[1]
            rows.append(row1)
            row2 = base_row.copy()
            row2[m] = other_chars[1]
            row2[n] = other_chars[0]
            rows.append(row2)
    return rows

def main():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    valid_rows = []
    for i in range(N):
        R_char = R[i]
        rows = generate_rows(R_char, N)
        if not rows:
            print("No")
            return
        valid_rows.append(rows)
    
    char_to_index = {'A': 0, 'B': 1, 'C': 2}
    C_chars = list(C)
    R_chars = list(R)
    
    def backtrack(current_row, topmost, counts, grid):
        if current_row == N:
            for j in range(N):
                if counts[j][0] != 1 or counts[j][1] != 1 or counts[j][2] != 1:
                    return None
            return grid.copy()
        for row in valid_rows[current_row]:
            new_topmost = topmost.copy()
            new_counts = [cnt.copy() for cnt in counts]
            valid = True
            for j in range(N):
                char = row[j]
                if char == '.':
                    continue
                if new_topmost[j] is None:
                    if char != C_chars[j]:
                        valid = False
                        break
                    new_topmost[j] = char
                    idx = char_to_index[char]
                    new_counts[j][idx] += 1
                    if new_counts[j][idx] > 1:
                        valid = False
                        break
                else:
                    idx = char_to_index[char]
                    if new_counts[j][idx] >= 1:
                        valid = False
                        break
                    new_counts[j][idx] += 1
                    if new_counts[j][idx] > 1:
                        valid = False
                        break
            if not valid:
                continue
            new_grid = grid + [row]
            result = backtrack(current_row + 1, new_topmost, new_counts, new_grid)
            if result is not None:
                return result
        return None
    
    initial_topmost = [None] * N
    initial_counts = [[0, 0, 0] for _ in range(N)]
    initial_grid = []
    solution = backtrack(0, initial_topmost, initial_counts, initial_grid)
    
    if solution:
        print("Yes")
        for row in solution:
            print(''.join(row))
    else:
        print("No")

if __name__ == "__main__":
    main()