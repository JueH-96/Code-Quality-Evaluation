def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Convert the input to integers
    xA, yA, xB, yB, xC, yC = map(int, data)
    
    # Compute squared lengths
    AB_sq = (xB - xA)**2 + (yB - yA)**2
    BC_sq = (xC - xB)**2 + (yC - yB)**2
    CA_sq = (xA - xC)**2 + (yA - yC)**2
    
    # Check for right angle using Pythagoras theorem
    if (AB_sq + BC_sq == CA_sq) or (AB_sq + CA_sq == BC_sq) or (BC_sq + CA_sq == AB_sq):
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()