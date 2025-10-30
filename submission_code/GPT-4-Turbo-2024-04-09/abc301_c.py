def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S = data[0]
    T = data[1]
    
    n = len(S)
    
    # Set of characters that can replace '@'
    atcoder_chars = set('atcoder')
    
    for i in range(n):
        s = S[i]
        t = T[i]
        
        if s == t:
            continue
        
        if s == '@' and t in atcoder_chars:
            continue
        
        if t == '@' and s in atcoder_chars:
            continue
        
        # If none of the above conditions are met, it's not possible to match
        print("No")
        return
    
    print("Yes")

if __name__ == "__main__":
    main()