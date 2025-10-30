import math

def main():
    import sys
    V1, V2, V3 = map(int, sys.stdin.readline().split())
    
    total = V1 + 2 * V2 + 3 * V3
    if total != 1029:
        print("No")
        return
    
    if V3 % 7 != 0:
        print("No")
        return
    
    v3_part = V3 // 7
    total_v2 = V2 + 14 * v3_part
    
    if total_v2 % 49 != 0:
        print("No")
        return
    
    s = total_v2 // 49
    
    found = False
    x_sol = None
    y_sol = None
    
    # Try all possible x in 0..7
    for x in range(0, 8):
        y_candidate = s - x
        if y_candidate < 0 or y_candidate > 7:
            continue
        if x * y_candidate == v3_part:
            found = True
            x_sol = x
            y_sol = y_candidate
            break
    
    if not found:
        # Try quadratic solution
        D = s * s - 4 * v3_part
        if D < 0:
            print("No")
            return
        sqrtD = int(math.isqrt(D))
        if sqrtD * sqrtD != D:
            print("No")
            return
        
        if (s + sqrtD) % 2 != 0 or (s - sqrtD) % 2 != 0:
            print("No")
            return
        
        x1 = (s + sqrtD) // 2
        x2 = (s - sqrtD) // 2
        candidates = [x1, x2]
        
        for xc in candidates:
            yc = s - xc
            if 0 <= xc <= 7 and 0 <= yc <= 7 and xc * yc == v3_part:
                found = True
                x_sol = xc
                y_sol = yc
                break
        
        if not found:
            print("No")
            return
    
    # At this point, x_sol and y_sol are valid
    x = x_sol
    y = y_sol
    a = 7 - x
    b = 7 - y
    
    print("Yes")
    print(f"0 0 0 0 {b} 0 {a} 0 0")

if __name__ == "__main__":
    main()