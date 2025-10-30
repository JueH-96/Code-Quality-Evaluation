def main():
    n = int(input().strip())
    S = input().strip()
    T = input().strip()
    
    for i in range(n):
        a = S[i]
        b = T[i]
        if a == b:
            continue
        if {a, b} == {'0', 'o'} or {a, b} == {'1', 'l'}:
            continue
        print("No")
        return
        
    print("Yes")

if __name__ == "__main__":
    main()