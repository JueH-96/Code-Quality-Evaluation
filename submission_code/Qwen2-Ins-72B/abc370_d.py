from bisect import bisect_left, bisect_right

def main():
    H, W, Q = map(int, input().split())
    walls = H * W
    rows = [[] for _ in range(H)]
    cols = [[] for _ in range(W)]
    for _ in range(Q):
        R, C = map(int, input().split())
        R -= 1
        C -= 1
        if rows[R] and cols[C]:
            if rows[R][-1] == C and cols[C][-1] == R:
                rows[R].pop()
                cols[C].pop()
                walls -= 1
            else:
                i = bisect_right(rows[R], C)
                j = bisect_right(cols[C], R)
                if i:
                    walls -= 1
                    rows[rows[R][i-1]].remove(C)
                if j:
                    walls -= 1
                    cols[cols[C][j-1]].remove(R)
                rows[R].append(C)
                cols[C].append(R)
        else:
            rows[R].append(C)
            cols[C].append(R)
            walls -= 1
    print(walls)

main()