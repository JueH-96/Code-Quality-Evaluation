def main():
    V1, V2, V3 = map(int, input().split())
    
    if V3 % 7 != 0:
        print("No")
        return
    
    V3_div_7 = V3 // 7
    
    for X in range(1, 8):
        if V3_div_7 % X == 0:
            Y = V3_div_7 // X
            if 1 <= Y <= 7:
                V2_calc = 49 * Y + 49 * X - 14 * X * Y
                V1_calc = 1029 - 98 * Y - 98 * X + 7 * X * Y
                
                if V2_calc == V2 and V1_calc == V1:
                    print("Yes")
                    print(f"0 0 0 0 {7-Y} 0 {7-X} 0 0")
                    return
    
    print("No")

if __name__ == "__main__":
    main()