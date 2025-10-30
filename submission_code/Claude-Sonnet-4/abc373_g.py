import math

def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def segments_intersect(p1, q1, p2, q2):
    # Check if line segment p1q1 and p2q2 intersect
    d1 = cross_product(p2, q2, p1)
    d2 = cross_product(p2, q2, q1)
    d3 = cross_product(p1, q1, p2)
    d4 = cross_product(p1, q1, q2)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    return False

def solve():
    n = int(input())
    
    P = []
    Q = []
    
    for i in range(n):
        a, b = map(int, input().split())
        P.append((a, b))
    
    for i in range(n):
        c, d = map(int, input().split())
        Q.append((c, d))
    
    # Try all possible permutations using backtracking
    def backtrack(assignment, used_q):
        if len(assignment) == n:
            return assignment[:]
        
        p_idx = len(assignment)
        
        for q_idx in range(n):
            if used_q[q_idx]:
                continue
            
            # Check if this assignment creates any intersections
            valid = True
            for prev_p in range(p_idx):
                prev_q = assignment[prev_p]
                if segments_intersect(P[prev_p], Q[prev_q], P[p_idx], Q[q_idx]):
                    valid = False
                    break
            
            if valid:
                assignment.append(q_idx)
                used_q[q_idx] = True
                
                result = backtrack(assignment, used_q)
                if result is not None:
                    return result
                
                assignment.pop()
                used_q[q_idx] = False
        
        return None
    
    result = backtrack([], [False] * n)
    
    if result is None:
        print(-1)
    else:
        # Convert to 1-indexed
        print(' '.join(str(x + 1) for x in result))

solve()