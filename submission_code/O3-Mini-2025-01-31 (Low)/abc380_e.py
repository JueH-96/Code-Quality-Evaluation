def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    
    # Initially, each cell has a unique color so we have n segments,
    # each representing just one cell.
    # segments is maintained as a list of lists: [l, r, color] in sorted order of l.
    segments = [[i, i, i] for i in range(1, n+1)]
    lefts = [seg[0] for seg in segments]
    
    # Global counts: count[i] gives the number of cells with color i.
    counts = [0] * (n+1)
    for i in range(1, n+1):
        counts[i] = 1

    out_lines = []
    
    # Utility: remove a segment at given index.
    def remove_segment(idx):
        seg = segments.pop(idx)
        lefts.pop(idx)
        return seg

    # Utility: insert a segment, maintaining sorted order by left.
    def insert_segment(seg):
        pos = bisect.bisect_left(lefts, seg[0])
        segments.insert(pos, seg)
        lefts.insert(pos, seg[0])
        return pos

    # Process queries.
    for _ in range(q):
        typ = next(it)
        if typ == "1":
            # Query type 1 x c: repaint the connected component containing cell x to color c.
            x = int(next(it))
            new_color = int(next(it))
            # Find the segment containing cell x.
            pos = bisect.bisect_right(lefts, x) - 1
            # Safety check (should always succeed in well-formed input).
            if pos < 0 or pos >= len(segments):
                continue
            seg = segments[pos]
            # If cell x is not in current segment, continue.
            if not(seg[0] <= x <= seg[1]):
                continue
            # If the segment already has the new color, do nothing.
            if seg[2] == new_color:
                continue
            # Update global counts:
            seg_length = seg[1] - seg[0] + 1
            counts[seg[2]] -= seg_length
            counts[new_color] += seg_length
            
            # Remove this segment and prepare a new segment for merging.
            new_seg = [seg[0], seg[1], new_color]
            remove_segment(pos)
            
            # Merge with left segment if it is adjacent and has the same color.
            while pos - 1 >= 0:
                left_seg = segments[pos - 1]
                if left_seg[2] == new_color and left_seg[1] + 1 == new_seg[0]:
                    new_seg[0] = left_seg[0]
                    # Since left_seg is already painted with new_color,
                    # no count update is needed.
                    remove_segment(pos - 1)
                    pos -= 1
                else:
                    break
            
            # Merge with right segment if it is adjacent and has the same color.
            while pos < len(segments):
                right_seg = segments[pos]
                if right_seg[2] == new_color and new_seg[1] + 1 == right_seg[0]:
                    new_seg[1] = right_seg[1]
                    remove_segment(pos)
                else:
                    break
            
            # Insert the new merged segment.
            insert_segment(new_seg)
            
        else:
            # Query type 2 c: Output the number of cells painted with color c.
            c_val = int(next(it))
            out_lines.append(str(counts[c_val]))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()