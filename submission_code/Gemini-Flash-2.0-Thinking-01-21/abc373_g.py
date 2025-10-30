import math

def orientation(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False

def solve():
    n = int(input())
    p_coords = []
    for _ in range(n):
        a, b = map(int, input().split())
        p_coords.append((a, b))
    q_coords = []
    for _ in range(n):
        c, d = map(int, input().split())
        q_coords.append((c, d))
    
    import itertools
    
    permutations = list(itertools.permutations(range(1, n + 1)))
    
    for r_permutation in permutations:
        segments = []
        intersects = False
        for i in range(n):
            segments.append(((p_coords[i][0], p_coords[i][1]), (q_coords[r_permutation[i]-1][0], q_coords[r_permutation[i]-1][1])))
            
        for i in range(n):
            for j in range(i + 1, n):
                if segments_intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                    intersects = True
                    break
            if intersects:
                break
                
        if not intersects:
            print(*(list(r_permutation)))
            return
            
    print("-1")

if __name__ == '__main__':
    solve()