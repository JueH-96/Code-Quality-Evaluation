def main():
    import sys
    V1, V2, V3 = map(int, sys.stdin.read().split())

    # Check if V1 = 1029 - 2*V2 - 3*V3
    if V1 != 1029 - 2*V2 - 3*V3:
        print("No")
        return
    
    # Check if V3 is divisible by 7
    if V3 % 7 != 0:
        print("No")
        return
    
    S = V3 // 7  # (7 - k2) * (7 - k3) = S
    
    # Find k2 and k3 such that (7 - k2) * (7 - k3) = S
    # k2 and k3 are integers between 0 and 7
    found = False
    for k2 in range(8):
        if (7 - k2) == 0:
            continue
        if S % (7 - k2) != 0:
            continue
        temp = S // (7 - k2)
        k3 = 7 - temp
        if 0 <= k3 <= 7:
            # Found a solution
            a1, b1, c1 = 0, 0, 0
            a2, b2, c2 = 0, k2, 0
            a3, b3, c3 = k3, 0, 0
            print("Yes")
            print(a1, b1, c1, a2, b2, c2, a3, b3, c3)
            return
    
    # If no solution found
    print("No")

if __name__ == "__main__":
    main()