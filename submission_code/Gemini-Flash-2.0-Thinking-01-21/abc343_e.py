import math

def solve():
    v1, v2, v3 = map(int, input().split())
    cube_side = 7
    cube_volume = cube_side ** 3
    
    if v1 != 3 * cube_volume - 2 * v2 - 3 * v3:
        print("No")
        return
        
    if v3 % 7 != 0:
        print("No")
        return
        
    if (v2 + 2 * v3) % 49 != 0:
        print("No")
        return
        
    p = v3 // 7
    s = (v2 + 2 * v3) // 49
    
    delta = s*s - 4*p
    if delta < 0:
        print("No")
        return
        
    sqrt_delta = math.sqrt(delta)
    if sqrt_delta != int(sqrt_delta):
        print("No")
        return
        
    sqrt_delta = int(sqrt_delta)
    lx1 = (s + sqrt_delta) / 2
    lx2 = (s - sqrt_delta) / 2
    
    lx_values = []
    if lx1 == int(lx1):
        lx_val = int(lx1)
        if 0 <= lx_val <= 7:
            lx_values.append(lx_val)
    if lx2 == int(lx2):
        lx_val = int(lx2)
        if 0 <= lx_val <= 7 and lx_val not in lx_values:
            lx_values.append(lx_val)
            
    for lx in lx_values:
        ly = s - lx
        if ly == int(ly):
            ly_val = int(ly)
            if 0 <= ly_val <= 7:
                dx = 7 - lx
                dy = 7 - ly
                print("Yes")
                print(f"0 0 0 0 {dy} 0 {dx} 0 0")
                return
                
    print("No")

if __name__ == '__main__':
    solve()