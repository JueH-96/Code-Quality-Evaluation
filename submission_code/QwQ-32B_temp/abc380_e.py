import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1

    cell_to_interval = [None] * (N + 2)  # 1-based indexing

    # color_data: for each color, stores intervals and starts
    color_data = defaultdict(lambda: {'intervals': [], 'starts': []})
    count = defaultdict(int)

    # Initialize each cell as its own interval
    for x in range(1, N + 1):
        interval = [x, x, x]  # [start, end, color]
        cell_to_interval[x] = interval
        c = x
        lst_intervals = color_data[c]['intervals']
        starts = color_data[c]['starts']
        s = interval[0]
        # Find insertion point
        pos = bisect.bisect_left(starts, s)
        lst_intervals.insert(pos, interval)
        starts.insert(pos, s)
        count[c] += 1

    for _ in range(Q):
        query_type = input[idx]
        idx += 1
        if query_type == '1':
            x = int(input[idx])
            idx += 1
            new_c = int(input[idx])
            idx += 1

            # Get the old interval
            old_interval = cell_to_interval[x]
            old_c = old_interval[2]

            # Remove from old_c's list
            lst_old = color_data[old_c]['intervals']
            starts_old = color_data[old_c]['starts']
            s_old = old_interval[0]
            pos = bisect.bisect_left(starts_old, s_old)
            del lst_old[pos]
            del starts_old[pos]
            # Update old_c's count
            count[old_c] -= (old_interval[1] - old_interval[0] + 1)

            # Change color to new_c
            old_interval[2] = new_c
            merged_intervals = [old_interval]
            merged_start = old_interval[0]
            merged_end = old_interval[1]
            merged_length = merged_end - merged_start + 1

            # Prepare to merge with new_c's intervals
            lst_new = color_data[new_c]['intervals']
            starts_new = color_data[new_c]['starts']

            # Collect all intervals to merge
            while True:
                s = merged_start
                pos = bisect.bisect_left(starts_new, s)
                merged = False

                # Check previous interval
                if pos > 0:
                    prev = lst_new[pos - 1]
                    if prev[1] >= merged_start - 1:
                        # Merge with previous
                        merged_start = prev[0]
                        merged_end = max(merged_end, prev[1])
                        merged_length = merged_end - merged_start + 1
                        merged_intervals.append(prev)
                        del lst_new[pos - 1]
                        del starts_new[pos - 1]
                        merged = True
                        continue

                # Check next interval
                if not merged and pos < len(lst_new):
                    next_ = lst_new[pos]
                    if next_[0] <= merged_end + 1:
                        merged_end = max(merged_end, next_[1])
                        merged_length = merged_end - merged_start + 1
                        merged_intervals.append(next_)
                        del lst_new[pos]
                        del starts_new[pos]
                        merged = True
                        continue

                if not merged:
                    break

            # Update count for new_c by subtracting merged intervals and adding new
            for interval in merged_intervals:
                count[new_c] -= (interval[1] - interval[0] + 1)
            count[new_c] += merged_length

            # Create merged interval and insert into new_c's list
            merged_interval = [merged_start, merged_end, new_c]
            pos = bisect.bisect_left(starts_new, merged_start)
            lst_new.insert(pos, merged_interval)
            starts_new.insert(pos, merged_start)

        else:
            c = int(input[idx])
            idx += 1
            print(count[c])

if __name__ == "__main__":
    main()