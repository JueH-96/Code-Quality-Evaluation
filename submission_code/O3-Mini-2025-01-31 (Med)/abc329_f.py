def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    q = int(next(it))
    
    # boxes[i] is a dictionary that maps color -> count of that color in box i.
    # Initialize boxes: box 1 to box n each have one ball of color C_i.
    boxes = [None] * (n + 1)
    for i in range(1, n + 1):
        c = int(next(it))
        boxes[i] = {c: 1}
    
    # Process each query.
    # For each query (a, b): move all balls from box a to box b, then print the distinct color count in box b.
    # We use a union-by-size style merge to maintain efficiency.
    # After each move, box a becomes empty.
    out_lines = []
    for _ in range(q):
        a = int(next(it))
        b = int(next(it))
        # Get the source and destination dictionaries.
        source = boxes[a]
        dest = boxes[b]
        
        # We want to merge source into dest (because the query forces the destination).
        # To optimize the merge, if the source is smaller (or equal) than dest,
        # we add source's entries into dest. Otherwise, we merge dest into source
        # and then assign the merged dictionary to b.
        if len(source) <= len(dest):
            for color, cnt in source.items():
                dest[color] = dest.get(color, 0) + cnt
            # After the merge, clear box a.
            boxes[a] = {}
            out_lines.append(str(len(dest)))
        else:
            for color, cnt in dest.items():
                source[color] = source.get(color, 0) + cnt
            # Now assign the merged dictionary to box b,
            # and clear box a.
            boxes[b] = source
            boxes[a] = {}
            out_lines.append(str(len(source)))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()