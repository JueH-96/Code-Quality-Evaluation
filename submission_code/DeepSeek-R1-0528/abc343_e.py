def main():
    import sys
    data = sys.stdin.read().split()
    V1 = int(data[0])
    V2 = int(data[1])
    V3 = int(data[2])
    
    total = V1 + 2 * V2 + 3 * V3
    if total != 1029:
        print("No")
        return
        
    if V2 == 0 and V3 == 0:
        print("Yes")
        print("0 0 0 7 0 0 14 0 0")
        return
        
    if V3 > 0:
        R = list(range(-6, 7))
    else:
        R = list(range(-12, 13))
        
    found = False
    for dx in R:
        for dy in R:
            for dz in R:
                for ex in R:
                    for ey in R:
                        for ez in R:
                            def get_f(a):
                                if a < 0:
                                    a = -a
                                if a >= 7:
                                    return 0
                                return 7 - a
                                
                            f_dx = get_f(dx)
                            f_dy = get_f(dy)
                            f_dz = get_f(dz)
                            I12 = f_dx * f_dy * f_dz
                            
                            f_ex = get_f(ex)
                            f_ey = get_f(ey)
                            f_ez = get_f(ez)
                            I13 = f_ex * f_ey * f_ez
                            
                            dx23 = ex - dx
                            dy23 = ey - dy
                            dz23 = ez - dz
                            f_dx23 = get_f(dx23)
                            f_dy23 = get_f(dy23)
                            f_dz23 = get_f(dz23)
                            I23 = f_dx23 * f_dy23 * f_dz23
                            
                            def get_overlap(a, b):
                                low = max(0, a, b)
                                high = min(7, a + 7, b + 7)
                                return max(0, high - low)
                                
                            ox = get_overlap(dx, ex)
                            oy = get_overlap(dy, ey)
                            oz = get_overlap(dz, ez)
                            I123 = ox * oy * oz
                            
                            if I123 == V3 and I12 + I13 + I23 - 3 * I123 == V2:
                                print("Yes")
                                print(f"0 0 0 {dx} {dy} {dz} {ex} {ey} {ez}")
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
            
    if not found:
        print("No")

if __name__ == "__main__":
    main()