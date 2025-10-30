# YOUR CODE HERE
def solve():
    n = int(input())
    chords = []
    for _ in range(n):
        chords.append(list(map(int, input().split())))

    def intersects(a, b, c, d, n):
        a -= 1
        b -= 1
        c -= 1
        d -= 1

        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c

        def cross(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        def on_segment(p, q, r):
            if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
                return True
            return False
        
        p1 = (a,0)
        p2 = (b,0)
        p3 = (c,1)
        p4 = (d,1)

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        if d1 == 0 and on_segment(p3, p1, p4):
            return True
        if d2 == 0 and on_segment(p3, p2, p4):
            return True
        if d3 == 0 and on_segment(p1, p3, p2):
            return True
        if d4 == 0 and on_segment(p1, p4, p2):
            return True
        
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if intersects(chords[i][0], chords[i][1], chords[j][0], chords[j][1], 2 * n):
                print("Yes")
                return

    print("No")

solve()