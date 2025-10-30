import sys

def main():
    data = sys.stdin.read().splitlines()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        S = data[idx].strip()
        idx += 1
        X = data[idx].strip()
        idx += 1
        Y = data[idx].strip()
        idx += 1
        
        a = X.count('0')
        b = X.count('1')
        c = Y.count('0')
        d = Y.count('1')
        
        if S == "":
            print("Yes")
        else:
            if a == c:
                print("Yes")
            else:
                denominator = a - c
                numerator = d - b
                if numerator % denominator != 0:
                    print("No")
                else:
                    k = numerator // denominator
                    if k <= 0:
                        print("No")
                    else:
                        if len(S) % k != 0:
                            print("No")
                        else:
                            m = len(S) // k
                            valid = True
                            for i in range(len(S)):
                                if S[i] != S[i % m]:
                                    valid = False
                                    break
                            print("Yes" if valid else "No")

if __name__ == "__main__":
    main()