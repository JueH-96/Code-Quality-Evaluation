def solve_cell_coloring(n, queries):
    colors = list(range(1, n + 1))  # Initially, cell i is painted with color i
    
    # Initialize size dictionary to track the count of each color
    size = {}
    for i in range(1, n + 1):
        size[i] = 1
    
    results = []
    
    for query in queries:
        if query[0] == 1:  # Recolor query
            _, x, c = query
            x -= 1  # Convert to 0-indexed
            old_color = colors[x]
            
            if old_color == c:  # No need to recolor if already the desired color
                continue
            
            # Find all connected cells with the same color using DFS
            recolor_cells = set()
            stack = [x]
            
            while stack:
                current = stack.pop()
                
                if current in recolor_cells:
                    continue
                
                recolor_cells.add(current)
                
                # Check left adjacent cell
                if current > 0 and colors[current - 1] == old_color:
                    stack.append(current - 1)
                
                # Check right adjacent cell
                if current < n - 1 and colors[current + 1] == old_color:
                    stack.append(current + 1)
            
            # Update color counts
            size[old_color] -= len(recolor_cells)
            size[c] = size.get(c, 0) + len(recolor_cells)
            
            # Recolor the cells
            for cell in recolor_cells:
                colors[cell] = c
        
        else:  # Count query
            _, c = query
            results.append(size.get(c, 0))
    
    return results

def main():
    n, q = map(int, input().split())
    queries = []
    
    for _ in range(q):
        query = list(map(int, input().split()))
        queries.append(query)
    
    results = solve_cell_coloring(n, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()