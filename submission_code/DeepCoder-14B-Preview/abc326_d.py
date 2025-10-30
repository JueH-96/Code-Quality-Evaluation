import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline().strip())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Convert R and C to lists for easier access
    R = list(R)
    C = list(C)

    # Step 1: Check if each row can have at least one possible column
    row_possible = [False] * N
    for i in range(N):
        for j in range(N):
            if C[j] == R[i]:
                row_possible[i] = True
                break
    if not all(row_possible):
        print("No")
        return

    # Step 2: Check if each column can have at least one possible row
    col_possible = [False] * N
    for j in range(N):
        for i in range(N):
            if R[i] == C[j]:
                col_possible[j] = True
                break
    if not all(col_possible):
        print("No")
        return

    # Step 3: Find bipartite matching
    # We'll model this as a bipartite graph and find a perfect matching
    # Using standard maximum bipartite matching algorithm
    graph = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if C[j] == R[i]:
                graph[i].append(j)

    # Implement Hopcroft-Karp algorithm for maximum bipartite matching
    def hopcroft_karp():
        pair_U = [-1] * N
        pair_V = [-1] * N
        dist = [0] * N

        def bfs():
            queue = deque()
            for u in range(N):
                if pair_U[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            dist_found = float('inf')
            while queue:
                u = queue.popleft()
                if dist[u] < dist_found:
                    for v in graph[u]:
                        if pair_V[v] == -1:
                            dist_found = dist[u] + 1
                        elif dist[pair_V[v]] == float('inf'):
                            dist[pair_V[v]] = dist[u] + 1
                            queue.append(pair_V[v])
            return dist_found != float('inf')

        def dfs(u):
            for v in graph[u]:
                if pair_V[v] == -1 or (dist[pair_V[v]] == dist[u] + 1 and dfs(pair_V[v])):
                    pair_U[u] = v
                    pair_V[v] = u
                    return True
            dist[u] = float('inf')
            return False

        result = 0
        while bfs():
            for u in range(N):
                if pair_U[u] == -1:
                    if dfs(u):
                        result += 1
        return result, pair_U

    max_matching, pair_U = hopcroft_karp()
    if max_matching != N:
        print("No")
        return

    # Extract the matching: for each row i, j_i is the matched column
    j_i = [0] * N
    for i in range(N):
        if pair_U[i] != -1:
            j_i[i] = pair_U[i]
        else:
            # This should not happen as we have a perfect matching
            print("No")
            return

    # Now, assign the first non-empty cells
    grid = [['.' for _ in range(N)] for _ in range(N)]
    for i in range(N):
        j = j_i[i]
        grid[i][j] = R[i]
        # Set all cells to the left of j to '.'
        for k in range(j):
            grid[i][k] = '.'

    # Now, we need to fill the remaining cells in each row
    # We'll try all possible ways to fill the remaining letters

    # We'll represent the grid as a list of lists
    # We'll try to fill each row's remaining letters and check column constraints

    # To manage the grid state, we'll create a deep copy for each possibility
    from copy import deepcopy

    # Function to check if all columns are valid
    def is_column_valid(grid, N):
        for j in range(N):
            counts = {'A':0, 'B':0, 'C':0}
            for i in range(N):
                cell = grid[i][j]
                if cell == 'A':
                    counts['A'] += 1
                elif cell == 'B':
                    counts['B'] += 1
                elif cell == 'C':
                    counts['C'] += 1
            if counts['A'] != 1 or counts['B'] != 1 or counts['C'] != 1:
                return False
        return True

    # Function to check if all rows are valid
    def is_row_valid(grid, N):
        for i in range(N):
            counts = {'A':0, 'B':0, 'C':0}
            for j in range(N):
                cell = grid[i][j]
                if cell == 'A':
                    counts['A'] += 1
                elif cell == 'B':
                    counts['B'] += 1
                elif cell == 'C':
                    counts['C'] += 1
            if counts['A'] != 1 or counts['B'] != 1 or counts['C'] != 1:
                return False
        return True

    # Try to fill the grid
    # We'll proceed row by row, filling each row's missing letters
    # and checking the column constraints as we go

    # We'll use a recursive backtracking approach
    # But given the small N, we can implement it iteratively

    # We'll make a deep copy of the grid for each possibility
    # and try to fill it

    # We'll proceed row by row
    # For each row, find the missing letters and available columns
    # Try all possible assignments of the missing letters to two available columns
    # and check if the column constraints are satisfied up to that point

    # We'll proceed row by row, starting from row 0 to row N-1
    # For each row, we'll try all possible valid assignments
    # and proceed to the next row if the current assignment is valid

    # We'll represent the current grid state and proceed iteratively

    # Initialize the grid
    current_grid = deepcopy(grid)

    # We'll process each row one by one
    # For each row, find the missing letters and available columns
    # Try all possible assignments and proceed

    # We'll use a list to represent the current grid state
    # and a list to track which columns are available for each row

    # We'll process each row in order
    # For each row, try all possible valid assignments
    # and proceed to the next row if valid

    # We'll use a queue for BFS approach
    # Each element in the queue represents a possible grid state
    # and the current row being processed

    from collections import deque

    queue = deque()
    queue.append(current_grid)

    # We'll also track the row we're processing
    # Each element in the queue is a tuple (grid_state, current_row)
    queue = deque()
    queue.append( (deepcopy(grid), 0) )

    while queue:
        grid_state, row = queue.popleft()

        if row == N:
            # All rows processed. Check if all columns are valid
            if is_column_valid(grid_state, N) and is_row_valid(grid_state, N):
                # Output the grid
                print("Yes")
                for r in grid_state:
                    print(''.join(r))
                return
            continue

        # Get the current row's data
        i = row
        present = set()
        for j in range(N):
            if grid_state[i][j] != '.':
                present.add(grid_state[i][j])
        missing = list( {'A', 'B', 'C'} - present )
        if len(missing) != 2:
            # This should not happen as we have exactly one missing letter
            continue

        # Find available columns in this row (after j_i[i] and not yet filled)
        available = []
        for j in range(N):
            if j > j_i[i] and grid_state[i][j] == '.':
                available.append(j)
        if len(available) < 2:
            continue  # Not enough space to fill the missing letters

        # Generate all possible ways to assign the two missing letters to two available columns
        from itertools import permutations
        for cols in permutations(available, 2):
            for perm in permutations(missing, 2):
                # Create a new grid state
                new_grid = deepcopy(grid_state)
                new_grid[i][cols[0]] = perm[0]
                new_grid[i][cols[1]] = perm[1]

                # Check if the column constraints are satisfied up to this row
                # We need to ensure that no column has more than one of the same letter
                # So, for each column, check if the new assignment causes a conflict
                valid = True
                for j in cols:
                    # Check the column j
                    counts = {'A':0, 'B':0, 'C':0}
                    for r in range(i+1):  # Only check up to current row
                        cell = new_grid[r][j]
                        if cell == 'A':
                            counts['A'] +=1
                        elif cell == 'B':
                            counts['B'] +=1
                        elif cell == 'C':
                            counts['C'] +=1
                        if counts['A'] > 1 or counts['B'] > 1 or counts['C'] > 1:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    # Add the new grid state to the queue for the next row
                    queue.append( (new_grid, row + 1) )

    # If no valid grid found
    print("No")

if __name__ == "__main__":
    main()