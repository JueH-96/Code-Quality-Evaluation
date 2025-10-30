def solve():
    s1s2 = input()
    t1t2 = input()
    
    s1 = s1s2[0]
    s2 = s1s2[1]
    t1 = t1t2[0]
    t2 = t1t2[1]

    def dist(p1, p2):
        points = ['A', 'B', 'C', 'D', 'E']
        idx1 = points.index(p1)
        idx2 = points.index(p2)
        
        diff = abs(idx1 - idx2)
        return min(diff, 5 - diff)
    
    if dist(s1, s2) == dist(t1, t2):
        print("Yes")
    else:
        print("No")

solve()