def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    xA, yA, xB, yB, xC, yC = map(int, data)
    
    def dist_sq(x1, y1, x2, y2):
        return (x1 - x2)**2 + (y1 - y2)**2
    
    # Calculate squared lengths of sides
    AB = dist_sq(xA, yA, xB, yB)
    BC = dist_sq(xB, yB, xC, yC)
    CA = dist_sq(xC, yC, xA, yA)
    
    # Check the Pythagorean theorem in any order
    if (AB + BC == CA) or (BC + CA == AB) or (CA + AB == BC):
        print("Yes")
    else:
        print("No")

# Do not remove the next line
main()