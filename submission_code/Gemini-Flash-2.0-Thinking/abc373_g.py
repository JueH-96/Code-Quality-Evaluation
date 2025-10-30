def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

def solve():
    n = int(input())
    p = []
    for _ in range(n):
        p.append(list(map(int, input().split())))
    q = []
    for _ in range(n):
        q.append(list(map(int, input().split())))

    def find_permutation(index, current_r):
        if index == n:
            return current_r

        p_i = p[index]

        for q_index in range(n):
            if q_index + 1 not in current_r:
                intersects = False
                q_r_i = q[q_index]
                for j in range(index):
                    p_j = p[j]
                    q_r_j_index = current_r[j] - 1
                    q_r_j = q[q_r_j_index]
                    if intersect(p_i, q_r_i, p_j, q_r_j):
                        intersects = True
                        break

                if not intersects:
                    result = find_permutation(index + 1, current_r + [q_index + 1])
                    if result:
                        return result

        return None

    result = find_permutation(0, [])
    if result:
        print(*result)
    else:
        print(-1)

solve()