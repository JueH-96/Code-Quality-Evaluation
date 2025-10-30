# YOUR CODE HERE
V1, V2, V3 = map(int, input().split())

if V1 + 2 * V2 + 3 * V3 != 3 * (7 ** 3):
    print("No")
else:
    if V3 % 7 != 0:
        print("No")
    else:
        k = V3 // 7
        numerator = V2 + 14 * k
        if numerator % 49 != 0:
            print("No")
        else:
            s = numerator // 49
            found = False
            a_found = 0
            b_found = 0
            for a in range(0, 8):
                if a == 0:
                    if k == 0:
                        b = s
                        if 0 <= b <= 7:
                            found = True
                            a_found = a
                            b_found = b
                            break
                else:
                    if k % a == 0:
                        b = k // a
                        if a + b == s and 0 <= b <= 7:
                            found = True
                            a_found = a
                            b_found = b
                            break
            if not found:
                print("No")
            else:
                dx = 7 - a_found
                dy = 7 - b_found
                print("Yes")
                print(f"0 0 0 {dx} 0 0 0 {dy} 0")