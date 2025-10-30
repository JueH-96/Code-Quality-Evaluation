import sys

def orientation(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

def segments_intersect(A, B, C, D):
    def ccw(p, q, r):
        val = orientation(p, q, r)
        if val > 0:
            return 1
        elif val < 0:
            return -1
        return 0

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    o1 = ccw(A, B, C)
    o2 = ccw(A, B, D)
    o3 = ccw(C, D, A)
    o4 = ccw(C, D, B)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if o1 == 0 and on_segment(A, C, B):
        return True
    if o2 == 0 and on_segment(A, D, B):
        return True
    if o3 == 0 and on_segment(C, A, D):
        return True
    if o4 == 0 and on_segment(C, B, D):
        return True

    return False

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1

    p_list = []
    for i in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        idx += 2
        p_list.append((x, y, i + 1))

    q_list = []
    for i in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        idx += 2
        q_list.append((x, y, i + 1))

    p_list_sorted = sorted(p_list, key=lambda x: (x[0], x[1]))
    q_list_sorted = sorted(q_list, key=lambda x: (x[0], x[1]))

    q_map = {q[2]: (q[0], q[1]) for q in q_list_sorted}

    R = {}
    used_Q = set()

    for i in range(len(p_list_sorted)):
        p = p_list_sorted[i]
        found = False
        for q in q_list_sorted:
            q_idx = q[2]
            if q_idx in used_Q:
                continue
            cross = False
            for j in range(i):
                p_prev = p_list_sorted[j]
                if p_prev[2] not in R:
                    continue
                q_prev_idx = R[p_prev[2]]
                A = (p_prev[0], p_prev[1])
                B = q_map[q_prev_idx]
                C = (p[0], p[1])
                D = (q[0], q[1])
                if segments_intersect(A, B, C, D):
                    cross = True
                    break
            if not cross:
                R[p[2]] = q_idx
                used_Q.add(q_idx)
                found = True
                break
        if not found:
            print(-1)
            return

    res = [R[i] for i in range(1, N + 1)]
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()