def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    # Read initial ball colors for the N boxes.
    colors = [int(next(it)) for _ in range(n)]
    # Each box is represented by a dictionary {color: count}.
    boxes = [{c: 1} for c in colors]

    # We'll collect the outputs for each query.
    output = []
    for _ in range(q):
        # Read a and b, convert to 0-indexed.
        a = int(next(it)) - 1
        b = int(next(it)) - 1

        # If source box (box a) is empty, nothing to move.
        if not boxes[a]:
            output.append(str(len(boxes[b])))
            continue

        # To optimize merging, merge the smaller dictionary into the larger.
        # Note: we must ensure that after merging, box b gets the union.
        if len(boxes[a]) > len(boxes[b]):
            # Swap the dictionaries so that boxes[a] becomes the smaller one.
            boxes[a], boxes[b] = boxes[b], boxes[a]
            
        # Now, merge the colors from box a (smaller) into box b.
        for color, cnt in boxes[a].items():
            boxes[b][color] = boxes[b].get(color, 0) + cnt

        # After merging, box a becomes empty.
        boxes[a].clear()
        
        # Append the number of distinct colors in box b.
        output.append(str(len(boxes[b])))

    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()