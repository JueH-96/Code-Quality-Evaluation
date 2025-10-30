# YOUR CODE HERE
import sys

# It is good practice for competitive programming in Python to increase the recursion limit.
# This is not strictly necessary for this specific solution since the `find`
# operation is implemented iteratively, but it's a useful habit.
# sys.setrecursionlimit(2 * 10**5 + 5)

def solve():
    """
    This function encapsulates the entire solution logic.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        
        N, Q = map(int, readline().split())

        # We use a Disjoint Set Union (DSU) data structure to manage the contiguous
        # components of same-colored cells. Each set in the DSU represents one
        # such component.
        #
        # We need to track not only the parent-child relationships for the sets,
        # but also additional properties for each component:
        # - its color
        # - its size (number of cells)
        # - its leftmost and rightmost cell indices
        #
        # We use 1-based indexing for cells 1 to N. Arrays are sized N+2 to
        # accommodate sentinels at index 0 and N+1, which simplifies boundary checks.

        parent = list(range(N + 2))
        # color[i] stores the color of the component whose root is i.
        color = list(range(N + 2)) 
        size = [1] * (N + 2)
        leftmost = list(range(N + 2))
        rightmost = list(range(N + 2))
        
        # Sentinels at index 0 and N+1 have size 0 as they don't represent actual cells.
        size[0] = 0
        size[N+1] = 0

        # `counts[c]` stores the total number of cells painted with color c.
        # Colors are from 1 to N, so an array of size N+1 is sufficient.
        counts = [0] * (N + 1)
        for i in range(1, N + 1):
            counts[i] = 1

        def find(i):
            """
            Finds the root of the set containing element i, with path compression.
            This is an iterative implementation to avoid Python's recursion depth limits.
            """
            path_to_root = []
            curr = i
            while curr != parent[curr]:
                path_to_root.append(curr)
                curr = parent[curr]
            root = curr
            for node in path_to_root:
                parent[node] = root
            return root

        def union(i, j):
            """
            Merges the sets containing elements i and j, using union by size.
            It also merges the component properties (size, leftmost, rightmost).
            """
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                # Union by size: attach smaller tree to the root of the larger tree
                if size[root_i] < size[root_j]:
                    root_i, root_j = root_j, root_i
                
                parent[root_j] = root_i
                size[root_i] += size[root_j]
                leftmost[root_i] = min(leftmost[root_i], leftmost[root_j])
                rightmost[root_i] = max(rightmost[root_i], rightmost[root_j])
                # The color of the merged component is handled by the calling logic
                # before the union is performed, as we only union components of the same color.

        # Process all Q queries
        for _ in range(Q):
            query_line = readline().split()
            query_type = int(query_line[0])
            
            if query_type == 1:
                # Repaint query: 1 x c
                x, c = int(query_line[1]), int(query_line[2])
                
                root_x = find(x)
                old_c = color[root_x]
                
                if old_c == c:
                    continue

                # 1. Update the counts of cells for the old and new colors.
                s = size[root_x]
                counts[old_c] -= s
                counts[c] += s
                
                # 2. Change the color of the entire component.
                color[root_x] = c
                
                # 3. Check if the repainted component can merge with its neighbors.
                # A merge is possible if a neighbor now has the same color `c`.
                
                # Check and merge with the left neighbor component.
                l = leftmost[root_x]
                if l > 1:
                    root_left = find(l - 1)
                    if color[root_left] == c:
                        union(x, l - 1)

                # The root of x's component might have changed after the left merge.
                # We must re-find it to get the updated rightmost boundary.
                root_x = find(x)
                
                # Check and merge with the right neighbor component.
                r = rightmost[root_x]
                if r < N:
                    root_right = find(r + 1)
                    if color[root_right] == c:
                        union(x, r + 1)

            else: # query_type == 2
                # Count query: 2 c
                c = int(query_line[1])
                print(counts[c])

    except (IOError, IndexError):
        # Graceful exit on empty input or other I/O errors.
        pass

# The problem requires reading from stdin and writing to stdout.
# The `if __name__ == "__main__":` block is standard practice in Python
# to make the script executable.
if __name__ == "__main__":
    solve()