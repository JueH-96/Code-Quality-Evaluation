import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        print("Yes")
        return
    N = int(line[0])
    M = int(line[1])
    # For each row x, maintain [lb, ub] for r_x
    rows = {}
    for _ in range(M):
        parts = data.readline().split()
        x = int(parts[0])
        y = int(parts[1])
        c = parts[2]
        if x not in rows:
            # r_x in [0..N]
            rows[x] = [0, N]
        lb, ub = rows[x]
        if c == 'B':
            # must have r_x >= y
            if y > lb:
                lb = y
        else:
            # 'W': must have r_x < y  => r_x <= y-1
            # y-1 might be negative but then ub becomes <0
            uy = y - 1
            if uy < ub:
                ub = uy
        rows[x][0], rows[x][1] = lb, ub

    # Now check each row for lb <= ub
    # And check monotonicity across sorted rows
    xs = sorted(rows.keys())
    prev_x = None
    prev_lb = prev_ub = None
    for x in xs:
        lb, ub = rows[x]
        if lb > ub:
            print("No")
            return
        if prev_x is not None:
            # need ub(prev) >= lb(curr)
            if prev_ub < lb:
                print("No")
                return
        prev_x = x
        prev_lb, prev_ub = lb, ub

    # If all checks passed
    print("Yes")

if __name__ == "__main__":
    main()