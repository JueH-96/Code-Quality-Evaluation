import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx]); idx +=1
    W = int(data[idx]); idx +=1
    Q = int(data[idx]); idx +=1

    rows = [[] for _ in range(H+1)]
    cols = [[] for _ in range(W+1)]

    for j in range(1, W+1):
        for i in range(1, H+1):
            bisect.insort(rows[i], j)
            bisect.insort(cols[j], i)

    total = H * W

    for _ in range(Q):
        R = int(data[idx]); idx +=1
        C = int(data[idx]); idx +=1

        destroyed = 0

        if C in rows[R]:
            pos = bisect.bisect_left(rows[R], C)
            if pos < len(rows[R]) and rows[R][pos] == C:
                del rows[R][pos]
                pos_col = bisect.bisect_left(cols[C], R)
                if pos_col < len(cols[C]) and cols[C][pos_col] == R:
                    del cols[C][pos_col]
            total -= 1
            destroyed += 1
        else:
            up_i = None
            idx_cols = bisect.bisect_left(cols[C], R)
            if idx_cols > 0:
                up_i = cols[C][idx_cols -1]
                if up_i < R:
                    pos = bisect.bisect_left(rows[up_i], C)
                    if pos < len(rows[up_i]) and rows[up_i][pos] == C:
                        del rows[up_i][pos]
                        pos_col = bisect.bisect_left(cols[C], up_i)
                        if pos_col < len(cols[C]) and cols[C][pos_col] == up_i:
                            del cols[C][pos_col]
                    total -= 1
                    destroyed += 1

            idx_cols = bisect.bisect_left(cols[C], R+1)
            if idx_cols < len(cols[C]):
                down_i = cols[C][idx_cols]
                if down_i > R:
                    pos = bisect.bisect_left(rows[down_i], C)
                    if pos < len(rows[down_i]) and rows[down_i][pos] == C:
                        del rows[down_i][pos]
                        pos_col = bisect.bisect_left(cols[C], down_i)
                        if pos_col < len(cols[C]) and cols[C][pos_col] == down_i:
                            del cols[C][pos_col]
                    total -= 1
                    destroyed += 1

            idx_rows = bisect.bisect_left(rows[R], C)
            if idx_rows > 0:
                left_j = rows[R][idx_rows -1]
                if left_j < C:
                    pos = idx_rows -1
                    del rows[R][pos]
                    pos_col = bisect.bisect_left(cols[left_j], R)
                    if pos_col < len(cols[left_j]) and cols[left_j][pos_col] == R:
                        del cols[left_j][pos_col]
                    total -= 1
                    destroyed += 1

            idx_rows = bisect.bisect_left(rows[R], C+1)
            if idx_rows < len(rows[R]):
                right_j = rows[R][idx_rows]
                if right_j > C:
                    pos = bisect.bisect_left(rows[R], right_j)
                    if pos < len(rows[R]) and rows[R][pos] == right_j:
                        del rows[R][pos]
                        pos_col = bisect.bisect_left(cols[right_j], R)
                        if pos_col < len(cols[right_j]) and cols[right_j][pos_col] == R:
                            del cols[right_j][pos_col]
                    total -= 1
                    destroyed += 1

        print(total)

if __name__ == "__main__":
    main()