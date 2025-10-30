import math

def main():
    V1, V2, V3 = map(int, input().split())
    total = V1 + 2 * V2 + 3 * V3
    if total != 3 * 7**3:
        print("No")
        return
    
    if V3 % 7 != 0:
        print("No")
        return
    k_sq = V3 // 7
    k = int(math.isqrt(k_sq))
    if k * k != k_sq or k < 0 or k > 7:
        print("No")
        return
    
    s = 7 - k
    expected_V2 = 14 * s * k
    if expected_V2 != V2:
        print("No")
        return
    
    expected_V1 = 1029 - 7 * k * (21 + s)
    if expected_V1 != V1:
        print("No")
        return
    
    a1, b1, c1 = 0, 0, 0
    a2, b2, c2 = 0, s, 0
    a3, b3, c3 = s, 0, 0
    print("Yes")
    print(a1, b1, c1, a2, b2, c2, a3, b3, c3)

if __name__ == "__main__":
    main()