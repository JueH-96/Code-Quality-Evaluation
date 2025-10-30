import sys

def readints():
    return list(map(int, sys.stdin.readline().split()))

def orientation(p, q, r):
    val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def segments_cross(a1, a2, b1, b2):
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)
    if o1 != o2 and o3 != o4:
        return True
    return False

def main():
    N = int(sys.stdin.readline())
    original_P = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    original_Q = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    sorted_P = sorted(original_P, key=lambda x: (x[0], x[1]))
    sorted_Q = sorted(original_Q, key=lambda x: (x[0], x[1]))
    
    used = [False] * N
    R = []
    pairs = []
    
    for p in sorted_P:
        found = False
        for j in range(N):
            if not used[j]:
                q = sorted_Q[j]
                cross = False
                for (pp, qq) in pairs:
                    if segments_cross(p, q, pp, qq):
                        cross = True
                        break
                if not cross:
                    R.append(j)
                    used[j] = True
                    pairs.append((p, q))
                    found = True
                    break
        if not found:
            print(-1)
            return
    
    original_Q_indices = [i+1 for i in range(N)]
    perm = [original_Q_indices[j] for j in R]
    print(' '.join(map(str, perm)))

if __name__ == "__main__":
    main()