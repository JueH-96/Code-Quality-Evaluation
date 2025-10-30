import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        S = sys.stdin.readline().strip()
        X = sys.stdin.readline().strip()
        Y = sys.stdin.readline().strip()
        
        count0_X = X.count('0')
        count1_X = len(X) - count0_X
        count0_Y = Y.count('0')
        count1_Y = len(Y) - count0_Y
        
        if X == Y:
            print("Yes")
        elif len(S) == 0:
            print("Yes")
        else:
            A = count1_X - count1_Y
            B = count0_Y - count0_X
            if A == 0 and B == 0:
                print("Yes")
            else:
                if A == 0 or B == 0:
                    print("No")
                elif ((A > 0) ^ (B > 0)):
                    print("No")
                else:
                    len_S = len(S)
                    numerator = B * len_S
                    if numerator % A != 0:
                        print("No")
                    else:
                        L = numerator // A
                        if A % B != 0:
                            print("No")
                        else:
                            k = A // B
                            if k <= 0:
                                print("No")
                            else:
                                T = S[:L]
                                if S == T * k:
                                    print("Yes")
                                else:
                                    print("No")

if __name__ == "__main__":
    main()