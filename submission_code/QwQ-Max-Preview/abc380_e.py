import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    intervals = []
    for i in range(1, N+1):
        intervals.append((i, i, i))
    color_counts = {}
    for i in range(1, N+1):
        color_counts[i] = 1

    starts = [interval[0] for interval in intervals]

    for _ in range(Q):
        query_type = int(data[ptr])
        ptr += 1
        if query_type == 1:
            x = int(data[ptr])
            ptr += 1
            c = int(data[ptr])
            ptr += 1

            idx = bisect.bisect_right(starts, x) - 1
            if idx < 0 or intervals[idx][1] < x:
                continue  # x not found in intervals (shouldn't happen)
            current = intervals[idx]
            start, end, old_color = current
            if old_color == c:
                continue

            del intervals[idx]
            del starts[idx]
            color_counts[old_color] -= (end - start + 1)

            new_interval = (start, end, c)
            insert_pos = bisect.bisect_left(starts, start)
            intervals.insert(insert_pos, new_interval)
            starts.insert(insert_pos, start)
            color_counts[c] = color_counts.get(c, 0) + (end - start + 1)

            merged = True
            while merged:
                merged = False
                # Check previous interval
                if insert_pos > 0:
                    prev = intervals[insert_pos - 1]
                    if prev[2] == c and prev[1] == start - 1:
                        new_start = prev[0]
                        new_end = end
                        new_c = c
                        # Remove previous and current
                        del intervals[insert_pos]
                        del starts[insert_pos]
                        del intervals[insert_pos - 1]
                        del starts[insert_pos - 1]
                        color_counts[c] -= (prev[1] - prev[0] + 1)
                        color_counts[c] -= (end - start + 1)
                        merged_interval = (new_start, new_end, new_c)
                        insert_pos = bisect.bisect_left(starts, new_start)
                        intervals.insert(insert_pos, merged_interval)
                        starts.insert(insert_pos, new_start)
                        color_counts[c] += (new_end - new_start + 1)
                        start, end, _ = merged_interval
                        merged = True
                # Check next interval
                if insert_pos < len(intervals) - 1:
                    next_int = intervals[insert_pos + 1]
                    if next_int[2] == c and next_int[0] == end + 1:
                        new_start = start
                        new_end = next_int[1]
                        new_c = c
                        # Remove current and next
                        del intervals[insert_pos + 1]
                        del starts[insert_pos + 1]
                        del intervals[insert_pos]
                        del starts[insert_pos]
                        color_counts[c] -= (end - start + 1)
                        color_counts[c] -= (next_int[1] - next_int[0] + 1)
                        merged_interval = (new_start, new_end, new_c)
                        insert_pos = bisect.bisect_left(starts, new_start)
                        intervals.insert(insert_pos, merged_interval)
                        starts.insert(insert_pos, new_start)
                        color_counts[c] += (new_end - new_start + 1)
                        start, end, _ = merged_interval
                        merged = True
        else:
            c = int(data[ptr])
            ptr += 1
            print(color_counts.get(c, 0))

if __name__ == "__main__":
    main()