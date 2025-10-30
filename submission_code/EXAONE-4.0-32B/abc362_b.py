def main():
    points = []
    for _ in range(3):
        x, y = map(int, input().split())
        points.append((x, y))
    
    A, B, C = points
    
    def sq_distance(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy
    
    ab_sq = sq_distance(A, B)
    bc_sq = sq_distance(B, C)
    ac_sq = sq_distance(A, C)
    
    sides = [ab_sq, bc_sq, ac_sq]
    sides.sort()
    
    if sides[0] + sides[1] == sides[2]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()