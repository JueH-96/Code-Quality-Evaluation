def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    k = sum(1 for a, b in zip(S, T) if a != b)
    
    if k == 0:
        print(0)
        return
    
    current_s = list(S)
    X = []
    
    for _ in range(k):
        candidates = [i for i in range(len(current_s)) if current_s[i] != T[i]]
        next_options = []
        for i in candidates:
            temp = current_s.copy()
            temp[i] = T[i]
            next_options.append(temp)
        min_option = min(next_options)
        X.append(''.join(min_option))
        current_s = min_option
    
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    main()