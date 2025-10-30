def solve():
    s1s2 = input()
    t1t2 = input()
    
    def dist(p1, p2):
        points = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        d = abs(points[p1] - points[p2])
        return min(d, 5 - d)

    if dist(s1s2[0], s1s2[1]) == dist(t1t2[0], t1t2[1]):
        print("Yes")
    else:
        print("No")

solve()