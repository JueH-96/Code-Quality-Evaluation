import sys

def solve():
    N = int(sys.stdin.readline())

    # RevAdj[i] stores the list of prerequisites for book i.
    # Using 1-based indexing for books (indices 1 to N). Index 0 is unused.
    RevAdj = [[] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        line_parts = list(map(int, sys.stdin.readline().split()))
        # line_parts[0] is C_i, the count of prerequisites.
        # Prerequisites themselves are line_parts[1:].
        prerequisites = line_parts[1:]
        RevAdj[i] = prerequisites

    # `path` will store the books to read (excluding book 1) in a valid reading order.
    path = []
    
    # `visited[i]` is True if book i has been pushed onto the DFS stack.
    visited = [False] * (N + 1)
    
    # `dfs_stack` stores tuples: (book_id, next_prereq_idx_to_visit).
    # `next_prereq_idx_to_visit` is an index into RevAdj[book_id].
    dfs_stack = []

    # Start DFS from book 1.
    # Mark book 1 as visited and push it onto the stack.
    dfs_stack.append((1, 0))
    visited[1] = True

    while dfs_stack:
        u, prereq_idx = dfs_stack[-1] # Peek at the top of the stack

        if prereq_idx < len(RevAdj[u]):
            # Current book `u` has more prerequisites to explore.
            p = RevAdj[u][prereq_idx] # Get the current prerequisite `p`.
            
            # Update stack top: next time `u` is processed, move to its next prerequisite.
            dfs_stack[-1] = (u, prereq_idx + 1)
            
            if not visited[p]:
                # If prerequisite `p` hasn't been visited, mark it and push for processing.
                visited[p] = True
                dfs_stack.append((p, 0))
        else:
            # All prerequisites of `u` have been processed (or attempts to process them made).
            # Pop `u` from the stack.
            processed_book, _ = dfs_stack.pop()
            
            # Add `processed_book` to `path` if it's not book 1.
            # Books are added to `path` in post-order. Since we explore from a book
            # to its prerequisites, prerequisites are processed and added to `path`
            # before the book that depends on them. This yields a topological sort.
            if processed_book != 1:
                path.append(processed_book)
                
    # Print the result: space-separated list of book numbers.
    # The problem constraints (C_1 >= 1, no cycles) imply `path` will not be empty.
    if path:
        sys.stdout.write(" ".join(map(str, path)) + "
")
    else:
        # This case is unlikely given problem constraints but handled for robustness.
        sys.stdout.write("
")

if __name__ == '__main__':
    solve()