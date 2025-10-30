def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S = data[0]
    T = data[1]
    
    n = len(S)
    X = []
    current = list(S)
    
    for i in range(n):
        if current[i] != T[i]:
            current[i] = T[i]
            X.append(''.join(current))
    
    print(len(X))
    for x in X:
        print(x)

if __name__ == "__main__":
    main()