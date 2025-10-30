def min_cost_bipartite_matching(cost_matrix):
    """
    Implementation of the Hungarian algorithm (Kuhn-Munkres) for minimum cost bipartite matching.
    
    Args:
        cost_matrix: An n x n matrix where cost_matrix[i][j] is the cost of assigning i to j.
        
    Returns:
        The total minimum cost and the assignment (a list where assignment[i] = j means i is assigned to j).
    """
    n = len(cost_matrix)
    
    # Make a deep copy of the cost matrix
    cost = [row[:] for row in cost_matrix]
    
    # Step 1: Subtract row minima
    for i in range(n):
        min_val = min(cost[i])
        for j in range(n):
            cost[i][j] -= min_val
    
    # Step 2: Subtract column minima
    for j in range(n):
        min_val = min(cost[i][j] for i in range(n))
        for i in range(n):
            cost[i][j] -= min_val
    
    # Step 3 and 4: Cover zeros with minimum number of lines
    row_assignment = [-1] * n
    col_assignment = [-1] * n
    
    # Find an initial assignment
    for i in range(n):
        for j in range(n):
            if cost[i][j] == 0 and col_assignment[j] == -1:
                row_assignment[i] = j
                col_assignment[j] = i
                break
    
    # Check if we have a complete assignment
    while -1 in row_assignment:
        # Find an unassigned row
        unassigned_row = row_assignment.index(-1)
        
        # Initialize marks for lines
        marked_rows = [False] * n
        marked_cols = [False] * n
        
        # Mark the unassigned row
        marked_rows[unassigned_row] = True
        
        # This array keeps track of the path we've taken
        path = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Mark alternating rows and columns
        marked = True
        while marked:
            marked = False
            for i in range(n):
                if marked_rows[i]:
                    for j in range(n):
                        if cost[i][j] == 0 and not marked_cols[j]:
                            path[i][j] = 1
                            marked_cols[j] = True
                            marked = True
            
            for j in range(n):
                if marked_cols[j]:
                    assigned_row = col_assignment[j]
                    if assigned_row != -1 and not marked_rows[assigned_row]:
                        path[assigned_row][j] = -1
                        marked_rows[assigned_row] = True
                        marked = True
        
        # Find the minimum uncovered value
        min_val = float('inf')
        for i in range(n):
            if marked_rows[i]:
                for j in range(n):
                    if not marked_cols[j]:
                        min_val = min(min_val, cost[i][j])
        
        # Adjust the cost matrix
        for i in range(n):
            for j in range(n):
                if marked_rows[i]:
                    cost[i][j] -= min_val
                if marked_cols[j]:
                    cost[i][j] += min_val
        
        # Try to find an augmenting path
        for i in range(n):
            for j in range(n):
                if row_assignment[i] == -1 and col_assignment[j] == -1 and cost[i][j] == 0:
                    row_assignment[i] = j
                    col_assignment[j] = i
                    break
    
    # Calculate the total cost using the original cost matrix
    total_cost = sum(cost_matrix[i][row_assignment[i]] for i in range(n))
    
    return total_cost, row_assignment

def main():
    # Read input
    n = int(input())
    A = list(map(int, input().split()))
    W = list(map(int, input().split()))
    
    # Create the cost matrix
    cost_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if A[i] == j + 1:  # A is 1-indexed, j is 0-indexed
                cost_matrix[i][j] = 0
            else:
                cost_matrix[i][j] = W[i]
    
    # Find the minimum cost assignment
    total_cost, _ = min_cost_bipartite_matching(cost_matrix)
    
    # Print the result
    print(total_cost)

if __name__ == "__main__":
    main()