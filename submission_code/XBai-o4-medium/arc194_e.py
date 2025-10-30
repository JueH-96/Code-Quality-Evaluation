def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    X = int(input[idx]); idx += 1
    Y = int(input[idx]); idx += 1
    S = input[idx]; idx += 1
    T = input[idx]
    
    if X + Y > N:
        if S == T:
            print("Yes")
        else:
            print("No")
        return
    
    current_S = list(S)
    
    i = 0
    while i < N:
        if current_S[i] == T[i]:
            i += 1
            continue
        else:
            if i + X + Y > N:
                print("No")
                return
            # Need to apply operation
            if current_S[i] == '0' and T[i] == '1':
                # Try Operation A
                valid = True
                for j in range(X):
                    if current_S[i + j] != '0':
                        valid = False
                        break
                if valid:
                    for j in range(Y):
                        if current_S[i + X + j] != '1':
                            valid = False
                            break
                if not valid:
                    print("No")
                    return
                # Apply Operation A
                for j in range(Y):
                    current_S[i + j] = '1'
                for j in range(X):
                    current_S[i + Y + j] = '0'
            else:
                # Try Operation B
                valid = True
                for j in range(Y):
                    if current_S[i + j] != '1':
                        valid = False
                        break
                if valid:
                    for j in range(X):
                        if current_S[i + Y + j] != '0':
                            valid = False
                            break
                if not valid:
                    print("No")
                    return
                # Apply Operation B
                for j in range(X):
                    current_S[i + j] = '0'
                for j in range(Y):
                    current_S[i + X + j] = '1'
            # After applying operation, move to next position
            i += 1
    
    if ''.join(current_S) == T:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()