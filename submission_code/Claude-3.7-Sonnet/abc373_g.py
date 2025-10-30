def solve():
    N = int(input())
    
    P = []
    for _ in range(N):
        a, b = map(int, input().split())
        P.append((a, b))
    
    Q = []
    for _ in range(N):
        c, d = map(int, input().split())
        Q.append((c, d))
    
    result = find_valid_permutation(N, P, Q)
    if result == [-1]:
        print(-1)
    else:
        print(" ".join(map(str, result)))

def find_valid_permutation(N, P, Q):
    # Initialize with an empty assignment
    assignment = []
    used_Qs = [False] * N
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 1 if val > 0 else 2  # val is never 0 with problem constraints
    
    def do_segments_intersect(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        return (o1 != o2 and o3 != o4)
    
    def backtrack(p_idx):
        if p_idx == N:
            return True
        
        for q_idx in range(N):
            if used_Qs[q_idx]:
                continue
            
            valid = True
            for i in range(p_idx):
                if do_segments_intersect(P[p_idx], Q[q_idx], P[i], Q[assignment[i]]):
                    valid = False
                    break
            
            if valid:
                assignment.append(q_idx)
                used_Qs[q_idx] = True
                
                if backtrack(p_idx + 1):
                    return True
                
                assignment.pop()
                used_Qs[q_idx] = False
        
        return False
    
    if backtrack(0):
        return [i + 1 for i in assignment]  # +1 for 1-indexed output
    else:
        return [-1]

solve()