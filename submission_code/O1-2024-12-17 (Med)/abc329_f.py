def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO helpers:
    #   We'll parse all inputs from input_data and then iterate through them.
    #   We'll store answers and print at the end for efficiency.
    
    # Parsing input
    N, Q = map(int, input_data[:2])
    colors = list(map(int, input_data[2:2+N]))
    queries = input_data[2+N:]
    
    # Prepare data structures
    # ds[i] will be a dict: color -> count of that color in box i
    # distinct[i] will be the number of distinct colors in box i
    ds = [{} for _ in range(N)]
    distinct = [0]*N
    
    # Initialize each box with its initial color
    for i in range(N):
        c = colors[i]
        ds[i][c] = 1
        distinct[i] = 1
    
    # Function to move contents from box a to box b
    # After this operation:
    #   box b should contain the union of old(a) U old(b)
    #   box a should become empty
    # We'll do "merge the smaller dictionary into the bigger dictionary" for efficiency
    def move(a, b):
        # If ds[a] has fewer or equal distinct colors, merge a into b directly
        if distinct[a] <= distinct[b]:
            # Merge ds[a] into ds[b]
            for color_a, count_a in ds[a].items():
                old_count_b = ds[b].get(color_a, 0)
                new_count_b = old_count_b + count_a
                ds[b][color_a] = new_count_b
                if old_count_b == 0:
                    distinct[b] += 1
            
            # Now a becomes empty
            ds[a].clear()
            distinct[a] = 0
        
        else:
            # ds[a] is bigger, so physically it's cheaper to merge b into a,
            # then swap them so that logically b ends up with the union.
            for color_b, count_b in ds[b].items():
                old_count_a = ds[a].get(color_b, 0)
                new_count_a = old_count_a + count_b
                ds[a][color_b] = new_count_a
                if old_count_a == 0:
                    distinct[a] += 1
            
            # b becomes empty
            ds[b].clear()
            distinct[b] = 0
            
            # Swap the dictionaries and distinct counts
            ds[a], ds[b] = ds[b], ds[a]
            distinct[a], distinct[b] = distinct[b], distinct[a]

    idx = 0
    output = []
    for _ in range(Q):
        a = int(queries[idx]) - 1  # convert to 0-based
        b = int(queries[idx+1]) - 1
        idx += 2
        
        move(a, b)
        output.append(str(distinct[b]))
    
    print("
".join(output))

# Call main() to ensure the solution runs
if __name__ == "__main__":
    main()