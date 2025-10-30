import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    results = []
    for _ in range(t):
        S = input[idx]
        idx += 1
        X = input[idx]
        idx += 1
        Y = input[idx]
        idx += 1
        m = len(X)
        n = len(Y)
        x0 = X.count('0')
        y0 = Y.count('0')
        len_S = len(S)
        A = x0 - y0
        B = (n - m) + A  # B = y1 - x1 = (n - y0) - (m - x0) = n - m + x0 - y0 = (n - m) + A
        # Check length equation
        possible = False
        if B == 0:
            if A == 0:
                possible = True
            else:
                possible = False
        else:
            numerator = A * len_S
            denominator = B
            if denominator == 0:
                possible = False
            else:
                if numerator % denominator != 0:
                    possible = False
                else:
                    len_T = numerator // denominator
                    if len_T < 0:
                        possible = False
                    else:
                        possible = True
        if not possible:
            results.append("No")
            continue
        # Now check string equation
        if m == n:
            # Compute c and d
            c = 0
            d = 0
            for i in range(m):
                if X[i] != Y[i]:
                    if X[i] == '0' and Y[i] == '1':
                        c += 1
                    else:
                        d += 1
            results.append("Yes")
        else:
            if A == 0 and (A * len_S == 0):
                # Check if len_T is 0
                if B != 0:
                    len_T = (A * len_S) // B
                else:
                    len_T = 0
                if len_T == 0:
                    if m > n:
                        extra = X[n:]
                        if all(ch == '1' for ch in extra):
                            possible = True
                        else:
                            possible = False
                    else:
                        extra = Y[m:]
                        if all(ch == '1' for ch in extra):
                            possible = True
                        else:
                            possible = False
                    if possible:
                        results.append("Yes")
                    else:
                        results.append("No")
                else:
                    results.append("No")
            else:
                results.append("No")
    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()