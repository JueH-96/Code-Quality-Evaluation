def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster output
    from sys import stdout
    
    # Parse the first line: N, Q
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse the colors
    colors = list(map(int, input_data[2:2+N]))
    
    # Queries start after the colors
    idx_queries = 2 + N
    
    # Each box will point to a (dict_of_color_counts, distinct_count)
    # We'll store these in a list "box_info", where box_info[i] is a reference
    # to the dictionary object for box i. Each dictionary object is:
    #   {"map": dict_of_color_counts, "distinct": number_of_distinct_colors}
    # Initially, box i has 1 ball of color colors[i], distinct count = 1.
    box_info = []
    for c in colors:
        d = {c: 1}  # color -> count
        box_info.append({"map": d, "distinct": 1})
    
    # We also keep a reference to an empty dictionary object for "clearing" boxes
    empty_dict_object = {"map": {}, "distinct": 0}
    
    # Helper function to merge smaller_dict into larger_dict
    def merge(smaller, larger):
        # Merging all color counts from smaller into larger
        s_map = smaller["map"]
        l_map = larger["map"]
        l_dist = larger["distinct"]
        
        for col, cnt in s_map.items():
            if col not in l_map:
                l_map[col] = cnt
                l_dist += 1
            else:
                l_map[col] += cnt
        
        # Now update larger's distinct
        larger["distinct"] = l_dist
        
        # Clear smaller
        s_map.clear()
        smaller["distinct"] = 0
    
    # Prepare output buffer
    out = []
    
    # Process each query
    ptr = idx_queries
    for _ in range(Q):
        a = int(input_data[ptr]) - 1  # convert to 0-based
        b = int(input_data[ptr+1]) - 1
        ptr += 2
        
        # If both boxes point to the same dictionary object, there's nothing to move
        if box_info[a] is box_info[b]:
            # Number of different colors in box b is just:
            out.append(str(box_info[b]["distinct"]))
            continue
        
        # We have two different dictionary objects
        dictA = box_info[a]
        dictB = box_info[b]
        
        # Merge the smaller map into the larger map
        if len(dictA["map"]) > len(dictB["map"]):
            # Swap so that dictA is always the smaller
            dictA, dictB = dictB, dictA
        
        # Merge dictA into dictB
        merge(dictA, dictB)
        
        # After the move, box a should be empty => point to empty_dict_object
        box_info[a] = empty_dict_object
        
        # Box b should point to the merged dictionary
        box_info[b] = dictB
        
        # The result is the distinct count in box b
        out.append(str(dictB["distinct"]))
    
    # Print all answers
    stdout.write("
".join(out) + "
")

# Let's call solve()
def main():
    solve()

if __name__ == "__main__":
    main()