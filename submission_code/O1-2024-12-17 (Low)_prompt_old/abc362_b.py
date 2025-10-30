def solve():
    import sys
    data = sys.stdin.read().strip().split()
    xA, yA, xB, yB, xC, yC = map(int, data)
    
    # Compute the squared lengths of the sides
    AB = (xB - xA)**2 + (yB - yA)**2
    BC = (xC - xB)**2 + (yC - yB)**2
    CA = (xA - xC)**2 + (yA - yC)**2
    
    # Check if any combination of two sides squared equals the remaining side squared
    if (AB + BC == CA) or (BC + CA == AB) or (CA + AB == BC):
        print("Yes")
    else:
        print("No")

# Call solve() to run
solve()