def is_right_triangle(xA, yA, xB, yB, xC, yC):
    a2 = (xB - xA)**2 + (yB - yA)**2
    b2 = (xC - xB)**2 + (yC - yB)**2
    c2 = (xA - xC)**2 + (yA - yC)**2
    sides = sorted([a2, b2, c2])
    if sides[0] + sides[1] == sides[2]:
        return "Yes"
    else:
        return "No"

xA, yA = map(int, input().split())
xB, yB = map(int, input().split())
xC, yC = map(int, input().split())
print(is_right_triangle(xA, yA, xB, yB, xC, yC))