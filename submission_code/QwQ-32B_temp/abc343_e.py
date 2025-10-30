def main():
    import sys
    V1, V2, V3 = map(int, sys.stdin.readline().split())
    
    # Check the first condition: V1 + 2V2 + 3V3 must equal 3*343
    total = V1 + 2 * V2 + 3 * V3
    if total != 3 * 343:
        print("No")
        return
    
    # Check if V3 is divisible by 7
    if V3 % 7 != 0:
        print("No")
        return
    
    K = V3 // 7
    found = False
    
    # Iterate over all possible Lx and Ly (0 <= Lx, Ly <=7)
    for Lx in range(0, 8):
        for Ly in range(0, 8):
            if Lx * Ly == K:
                v2_candidate = 49 * (Lx + Ly) - 14 * Lx * Ly
                if v2_candidate == V2:
                    # Found valid Lx and Ly
                    a1, b1, c1 = 0, 0, 0
                    a2, b2, c2 = 0, 7 - Ly, 0
                    a3, b3, c3 = 7 - Lx, 0, 0
                    print("Yes")
                    print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
                    return
    
    # If no solution found in the considered configuration
    print("No")

if __name__ == "__main__":
    main()