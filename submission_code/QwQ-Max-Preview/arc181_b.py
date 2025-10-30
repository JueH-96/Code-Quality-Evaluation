import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    results = []
    for _ in range(t):
        S = input[idx]
        X = input[idx+1]
        Y = input[idx+2]
        idx +=3
        
        lenX = len(X)
        lenY = len(Y)
        if X == Y:
            results.append("Yes")
            continue
        if lenX != lenY:
            # Check if T can be empty
            cnt_x0 = X.count('0')
            cnt_y0 = Y.count('0')
            if (S == '' or cnt_x0 == cnt_y0):
                results.append("Yes")
                continue
            # Check if T can be S
            if (S == '' or lenX == lenY):
                results.append("Yes")
                continue
            results.append("No")
            continue
        # Now lenX == lenY and X != Y
        # Check if all differing positions require T to be S
        # If any position where X has 0 and Y has 1, or vice versa, then T must be S
        # So after substitution, the concatenations are S * lenX and S * lenY (which are equal)
        # So check if S is empty or lenX == lenY (which they are)
        has_conflict = False
        for x_char, y_char in zip(X, Y):
            if x_char != y_char:
                if (x_char == '0' and y_char == '1') or (x_char == '1' and y_char == '0'):
                    has_conflict = True
                    break
        if has_conflict:
            # T must be S, check if S * lenX == S * lenY (which is true since lenX == lenY)
            results.append("Yes" if S == '' or lenX == lenY else "No")
        else:
            # All differing positions are 0 vs 0 or 1 vs 1. So substitution sequences are same after replacing 1's with T
            # So the concatenations are same for any T
            results.append("Yes")
    print('
'.join(results))

if __name__ == '__main__':
    main()