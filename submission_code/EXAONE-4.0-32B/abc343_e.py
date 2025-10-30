def main():
    data = input().split()
    V1 = int(data[0])
    V2 = int(data[1])
    V3 = int(data[2])
    
    total = V1 + 2 * V2 + 3 * V3
    if total != 1029:
        print("No")
        return
        
    if V1 == V2:
        if V1 % 49 == 0 and V3 % 49 == 0:
            m = V1 // 49
            n = V3 // 49
            if 0 <= m <= 7 and 0 <= n <= 7 and m + n == 7:
                print("Yes")
                print(f"0 0 0 {m} 0 0 {m} 0 0")
                return
                
    if V3 % 7 == 0:
        k = V3 // 7
        found = False
        a_val = None
        b_val = None
        for a in range(0, 8):
            if k == 0:
                if a == 0:
                    if V2 % 49 == 0:
                        b_candidate = V2 // 49
                        if 0 <= b_candidate <= 7:
                            found = True
                            a_val = 0
                            b_val = b_candidate
                            break
                else:
                    if V2 == 49 * a:
                        found = True
                        a_val = a
                        b_val = 0
                        break
            else:
                if a == 0:
                    continue
                if k % a == 0:
                    b = k // a
                    if b < 0 or b > 7:
                        continue
                    if V2 == 49 * (a + b) - 2 * V3:
                        found = True
                        a_val = a
                        b_val = b
                        break
                        
        if found:
            s = 7 - a_val
            t = 7 - b_val
            print("Yes")
            print(f"0 0 0 0 {s} 0 {t} 0 0")
            return
            
    print("No")

if __name__ == "__main__":
    main()