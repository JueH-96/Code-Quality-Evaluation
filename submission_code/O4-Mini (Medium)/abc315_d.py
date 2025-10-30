import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]

    # Prepare counts
    # row_letter_count[i][l]: count of letter l in row i
    # col_letter_count[j][l]: count of letter l in col j
    row_letter_count = [ [0]*26 for _ in range(H) ]
    col_letter_count = [ [0]*26 for _ in range(W) ]
    for i in range(H):
        row = grid[i]
        for j, ch in enumerate(row):
            idx = ord(ch) - 97
            row_letter_count[i][idx] += 1
            col_letter_count[j][idx] += 1

    # Remaining counts and distinct counts
    row_rem_cnt = [W]*H
    col_rem_cnt = [H]*W
    row_distinct = [0]*H
    col_distinct = [0]*W
    for i in range(H):
        d = 0
        for l in range(26):
            if row_letter_count[i][l] > 0:
                d += 1
        row_distinct[i] = d
    for j in range(W):
        d = 0
        for l in range(26):
            if col_letter_count[j][l] > 0:
                d += 1
        col_distinct[j] = d

    # Removal flags
    removed_row = [False]*H
    removed_col = [False]*W

    # Queues for rows/cols to remove next iteration
    row_queue = []
    col_queue = []
    row_in_q = [False]*H
    col_in_q = [False]*W

    # Initialize
    for i in range(H):
        if row_rem_cnt[i] >= 2 and row_distinct[i] == 1:
            row_queue.append(i)
            row_in_q[i] = True
    for j in range(W):
        if col_rem_cnt[j] >= 2 and col_distinct[j] == 1:
            col_queue.append(j)
            col_in_q[j] = True

    # Total remaining cookies
    total_rem = H * W

    # Process rounds
    while row_queue or col_queue:
        # take current to_remove lists
        to_rows = row_queue
        to_cols = col_queue
        row_queue = []
        col_queue = []
        # Note: we do not clear in_q flags, but removed flags prevent re-enqueue

        # Remove rows
        for i in to_rows:
            if removed_row[i]:
                continue
            # remove whole row i
            removed_row[i] = True
            # subtract remaining cells in row i from total
            total_rem -= row_rem_cnt[i]
            # after removal, row has 0 rem
            row_rem_cnt[i] = 0
            # update columns
            row_str = grid[i]
            for j in range(W):
                if removed_col[j]:
                    continue
                # cell (i,j) is now removed
                ch = row_str[j]
                idx = ord(ch) - 97
                # decrement column letter count
                cnt_old = col_letter_count[j][idx]
                cnt_new = cnt_old - 1
                col_letter_count[j][idx] = cnt_new
                if cnt_new == 0:
                    # one distinct letter gone
                    col_distinct[j] -= 1
                # decrement column rem count
                col_rem_cnt[j] -= 1
                # check if column j now qualifies
                if (not removed_col[j]
                        and not col_in_q[j]
                        and col_rem_cnt[j] >= 2
                        and col_distinct[j] == 1):
                    col_in_q[j] = True
                    col_queue.append(j)

        # Remove columns
        for j in to_cols:
            if removed_col[j]:
                continue
            # remove whole column j
            removed_col[j] = True
            total_rem -= col_rem_cnt[j]
            col_rem_cnt[j] = 0
            # update rows
            for i in range(H):
                if removed_row[i]:
                    continue
                # cell (i,j) is now removed
                ch = grid[i][j]
                idx = ord(ch) - 97
                cnt_old = row_letter_count[i][idx]
                cnt_new = cnt_old - 1
                row_letter_count[i][idx] = cnt_new
                if cnt_new == 0:
                    row_distinct[i] -= 1
                row_rem_cnt[i] -= 1
                # check if row i now qualifies
                if (not removed_row[i]
                        and not row_in_q[i]
                        and row_rem_cnt[i] >= 2
                        and row_distinct[i] == 1):
                    row_in_q[i] = True
                    row_queue.append(i)

    # Print result
    print(total_rem)

if __name__ == "__main__":
    main()