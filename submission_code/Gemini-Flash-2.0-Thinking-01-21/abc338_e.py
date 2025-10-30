import math

def solve():
    n = int(input())
    chords = []
    for _ in range(n):
        u, v = map(int, input().split())
        chords.append((u, v))
    
    points = {}
    for i in range(1, 2 * n + 1):
        angle = 2 * math.pi * (i - 1) / (2 * n)
        points[i] = (math.cos(angle), math.sin(angle))
        
    def orientation(p1_index, p2_index, p3_index):
        p1 = points[p1_index]
        p2 = points[p2_index]
        p3 = points[p3_index]
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        
    has_intersection = False
    for i in range(n):
        for j in range(i + 1, n):
            chord1 = chords[i]
            chord2 = chords[j]
            a1, b1 = chord1
            a2, b2 = chord2
            
            o1 = orientation(a1, b1, a2)
            o2 = orientation(a1, b1, b2)
            o3 = orientation(a2, b2, a1)
            o4 = orientation(a2, b2, b1)
            
            if (o1 * o2 < 0) and (o3 * o4 < 0):
                has_intersection = True
                break
        if has_intersection:
            break
            
    if has_intersection:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()