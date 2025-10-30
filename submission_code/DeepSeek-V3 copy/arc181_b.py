def can_equal(S, X, Y):
    # Calculate the number of 0s and 1s in X and Y
    x0 = X.count('0')
    x1 = len(X) - x0
    y0 = Y.count('0')
    y1 = len(Y) - y0
    
    # The total length of f(S,T,X) is x0 * len(S) + x1 * len(T)
    # Similarly for f(S,T,Y): y0 * len(S) + y1 * len(T)
    # For them to be equal, x0 * len(S) + x1 * len(T) = y0 * len(S) + y1 * len(T)
    # Rearranged: (x0 - y0) * len(S) = (y1 - x1) * len(T)
    
    # Let a = x0 - y0, b = y1 - x1
    a = x0 - y0
    b = y1 - x1
    
    # If b == 0, then a * len(S) must be 0
    if b == 0:
        if a == 0:
            return True
        else:
            return False
    else:
        # len(T) = (a * len(S)) / b
        # len(T) must be a non-negative integer
        if (a * len(S)) % b != 0:
            return False
        len_T = (a * len(S)) // b
        if len_T < 0:
            return False
        return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    t = int(data[0])
    index = 1
    for _ in range(t):
        S = data[index]
        X = data[index+1]
        Y = data[index+2]
        index += 3
        if can_equal(S, X, Y):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()