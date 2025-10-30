def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    xA, yA, xB, yB, xC, yC = map(int, data)
    
    # Calculate squared lengths of sides
    dAB = (xB - xA)**2 + (yB - yA)**2
    dBC = (xC - xB)**2 + (yC - yB)**2
    dCA = (xA - xC)**2 + (yA - yC)**2
    
    # Sort the squared lengths
    sides = sorted([dAB, dBC, dCA])
    
    # Check the Pythagorean theorem
    if sides[2] == sides[0] + sides[1]:
        print("Yes")
    else:
        print("No")

# Do not remove:
main()