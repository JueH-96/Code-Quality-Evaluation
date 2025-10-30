import sys
from itertools import permutations, combinations

def main():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Precompute all possible row options for each row
    row_options = []
    for i in range(N):
        first_char = R[i]
        possible_perms = []
        # Generate all permutations of A, B, C that start with first_char
        for p in permutations(['A', 'B', 'C']):
            if p[0] == first_char:
                possible_perms.append(p)
        # Generate all possible column triplets (j < k < l)
        col_triplets = list(combinations(range(N), 3))
        # Create row options for each permutation and column triplet
        options = []
        for perm in possible_perms:
            for triplet in col_triplets:
                j, k, l = triplet
                row = ['.'] * N
                row[j] = perm[0]
                row[k] = perm[1]
                row[l] = perm[2]
                options.append(''.join(row))
        row_options.append(options)
    
    # Initialize column states: top character and counts
    initial_top = [None] * N
    initial_counts = [{'A': 0, 'B': 0, 'C': 0} for _ in range(N)]

    solution = None

    def backtrack(current_row, top_chars, counts, current_solution):
        if current_row == N:
            # Check all columns have exactly one of each character
            for j in range(N):
                if counts[j]['A'] != 1 or counts[j]['B'] != 1 or counts[j]['C'] != 1:
                    return None
            return current_solution
        # Try each possible row option for the current row
        for row in row_options[current_row]:
            new_top = list(top_chars)
            new_counts = [dict(c) for c in counts]
            valid = True
            for j in range(N):
                c = row[j]
                if c == '.':
                    continue
                if new_top[j] is None:
                    if c != C[j]:
                        valid = False
                        break
                    new_top[j] = c
                    new_counts[j][c] += 1
                    if new_counts[j][c] > 1:
                        valid = False
                        break
                else:
                    if c == new_top[j]:
                        valid = False
                        break
                    new_counts[j][c] += 1
                    if new_counts[j][c] > 1:
                        valid = False
                        break
            if valid:
                new_solution = current_solution + [row]
                result = backtrack(current_row + 1, new_top, new_counts, new_solution)
                if result is not None:
                    return result
        return None

    solution = backtrack(0, initial_top, initial_counts, [])

    if solution:
        print("Yes")
        for line in solution:
            print(line)
    else:
        print("No")

if __name__ == '__main__':
    main()