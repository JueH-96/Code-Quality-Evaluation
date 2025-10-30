import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    
    queries = []
    index = 3
    for _ in range(Q):
        R = int(data[index])
        C = int(data[index + 1])
        queries.append((R, C))
        index += 2
    
    # Initialize rows and columns
    rows = {}
    for r in range(1, H+1):
        rows[r] = list(range(1, W+1))
    
    columns = {}
    for c in range(1, W+1):
        columns[c] = list(range(1, H+1))
    
    for r, c in queries:
        # Check if there's a wall at (r, c)
        pos_c = bisect.bisect_left(rows[r], c)
        if pos_c < len(rows[r]) and rows[r][pos_c] == c:
            # Wall exists at (r, c), destroy it
            rows[r].pop(pos_c)
            pos_r = bisect.bisect_left(columns[c], r)
            if pos_r < len(columns[c]) and columns[c][pos_r] == r:
                columns[c].pop(pos_r)
        else:
            # No wall at (r, c), destroy nearest walls in four directions
            # Up in column c
            pos = bisect.bisect_left(columns[c], r)
            if pos > 0:
                wall_above = columns[c][pos-1]
                # Remove wall_above from columns[c]
                columns[c].pop(pos-1)
                # Remove c from rows[wall_above]
                pos_remove = bisect.bisect_left(rows[wall_above], c)
                if pos_remove < len(rows[wall_above]) and rows[wall_above][pos_remove] == c:
                    rows[wall_above].pop(pos_remove)
            # Down in column c
            if pos < len(columns[c]) and columns[c][pos] > r:
                wall_below = columns[c][pos]
                # Remove wall_below from columns[c]
                columns[c].pop(pos)
                # Remove c from rows[wall_below]
                pos_remove = bisect.bisect_left(rows[wall_below], c)
                if pos_remove < len(rows[wall_below]) and rows[wall_below][pos_remove] == c:
                    rows[wall_below].pop(pos_remove)
            # Left in row r
            pos = bisect.bisect_left(rows[r], c)
            if pos > 0:
                wall_left = rows[r][pos-1]
                # Remove wall_left from rows[r]
                rows[r].pop(pos-1)
                # Remove r from columns[wall_left]
                pos_remove = bisect.bisect_left(columns[wall_left], r)
                if pos_remove < len(columns[wall_left]) and columns[wall_left][pos_remove] == r:
                    columns[wall_left].pop(pos_remove)
            # Right in row r
            if pos < len(rows[r]) and rows[r][pos] > c:
                wall_right = rows[r][pos]
                # Remove wall_right from rows[r]
                rows[r].pop(pos)
                # Remove r from columns[wall_right]
                pos_remove = bisect.bisect_left(columns[wall_right], r)
                if pos_remove < len(columns[wall_right]) and columns[wall_right][pos_remove] == r:
                    columns[wall_right].pop(pos_remove)
    
    # Count remaining walls
    total_walls = 0
    for r in rows:
        total_walls += len(rows[r])
    print(total_walls)

if __name__ == '__main__':
    main()