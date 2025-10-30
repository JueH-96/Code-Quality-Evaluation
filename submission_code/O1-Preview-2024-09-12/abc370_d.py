# YOUR CODE HERE
import sys
import threading
import bisect

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, Q = map(int, sys.stdin.readline().split())

    # For each row and column, maintain sorted list of intervals where walls exist
    from collections import defaultdict

    row_intervals = [ [(1, W)] for _ in range(H + 1) ]  # index from 1 to H
    col_intervals = [ [(1, H)] for _ in range(W + 1) ]  # index from 1 to W

    def find_interval(intervals, point):
        idx = bisect.bisect_right(intervals, (point, float('inf')))
        if idx == 0:
            return None, None
        idx -= 1
        start, end = intervals[idx]
        if start <= point <= end:
            return idx, (start, end)
        else:
            return None, None

    def remove_point(intervals, idx, interval, point):
        start, end = interval
        intervals.pop(idx)
        if start == end == point:
            pass  # interval removed
        elif start == point:
            intervals.insert(idx, (point + 1, end))
        elif end == point:
            intervals.insert(idx, (start, point - 1))
        else:
            # Split into two intervals
            intervals.insert(idx, (start, point - 1))
            intervals.insert(idx + 1, (point + 1, end))

    def find_and_remove_adjacent(intervals, point, direction):
        if direction == 'left' or direction == 'up':
            key_point = point - 1
            idx = bisect.bisect_right(intervals, (key_point, float('inf')))
            if idx == 0:
                return False
            idx -= 1
            start, end = intervals[idx]
            if start <= key_point <= end:
                remove_point(intervals, idx, (start, end), key_point)
                return True
            else:
                return False
        elif direction == 'right' or direction == 'down':
            key_point = point + 1
            idx = bisect.bisect_left(intervals, (key_point, -float('inf')))
            if idx == len(intervals):
                return False
            start, end = intervals[idx]
            if start <= key_point <= end:
                remove_point(intervals, idx, (start, end), key_point)
                return True
            else:
                return False
        else:
            return False  # Should not reach here

    # Process queries
    for _ in range(Q):
        R_q, C_q = map(int, sys.stdin.readline().split())

        # First, check if there is a wall at (R_q, C_q)
        idx_row, interval_row = find_interval(row_intervals[R_q], C_q)
        if idx_row is not None:
            # There is a wall at (R_q, C_q), remove it
            remove_point(row_intervals[R_q], idx_row, interval_row, C_q)
            # Also remove from column intervals
            idx_col, interval_col = find_interval(col_intervals[C_q], R_q)
            remove_point(col_intervals[C_q], idx_col, interval_col, R_q)
        else:
            # No wall at (R_q, C_q), need to destroy walls in four directions
            # Up
            idx_col = bisect.bisect_left(col_intervals[C_q], (R_q, -float('inf')))
            # Find predecessor interval
            destroyed = False
            if idx_col > 0:
                idx_col -=1
                start, end = col_intervals[C_q][idx_col]
                if start <= R_q - 1 <= end:
                    # Wall at (R_q - 1, C_q)
                    remove_point(col_intervals[C_q], idx_col, (start, end), R_q - 1)
                    # Also remove from row intervals
                    idx_row, interval_row = find_interval(row_intervals[R_q - 1], C_q)
                    remove_point(row_intervals[R_q - 1], idx_row, interval_row, C_q)
                    destroyed = True
            # Down
            idx_col = bisect.bisect_right(col_intervals[C_q], (R_q, float('inf')))
            if idx_col < len(col_intervals[C_q]):
                start, end = col_intervals[C_q][idx_col]
                if start <= R_q + 1 <= end:
                    # Wall at (R_q + 1, C_q)
                    remove_point(col_intervals[C_q], idx_col, (start, end), R_q + 1)
                    # Also remove from row intervals
                    idx_row, interval_row = find_interval(row_intervals[R_q + 1], C_q)
                    remove_point(row_intervals[R_q + 1], idx_row, interval_row, C_q)
                    destroyed = True
            # Left
            idx_row = bisect.bisect_left(row_intervals[R_q], (C_q, -float('inf')))
            if idx_row > 0:
                idx_row -=1
                start, end = row_intervals[R_q][idx_row]
                if start <= C_q - 1 <= end:
                    # Wall at (R_q, C_q - 1)
                    remove_point(row_intervals[R_q], idx_row, (start, end), C_q - 1)
                    # Also remove from column intervals
                    idx_col, interval_col = find_interval(col_intervals[C_q - 1], R_q)
                    remove_point(col_intervals[C_q - 1], idx_col, interval_col, R_q)
                    destroyed = True
            # Right
            idx_row = bisect.bisect_right(row_intervals[R_q], (C_q, float('inf')))
            if idx_row < len(row_intervals[R_q]):
                start, end = row_intervals[R_q][idx_row]
                if start <= C_q +1 <= end:
                    # Wall at (R_q, C_q +1)
                    remove_point(row_intervals[R_q], idx_row, (start, end), C_q +1)
                    # Also remove from column intervals
                    idx_col, interval_col = find_interval(col_intervals[C_q +1], R_q)
                    remove_point(col_intervals[C_q +1], idx_col, interval_col, R_q)
                    destroyed = True
            # If top row or bottom row, or leftmost or rightmost column
            # walls may not exist, so we don't need to do anything

    # After processing all queries, count remaining walls
    total_walls = 0
    for i in range(1, H +1):
        intervals = row_intervals[i]
        for start, end in intervals:
            total_walls += end - start +1
    print(total_walls)

if __name__ == "__main__":
    threading.Thread(target=main).start()