def main():
    n = int(input().strip())
    S = input().strip()
    T = input().strip()
    
    for i in range(n):
        if S[i] == T[i]:
            continue
        pair = {S[i], T[i]}
        if pair == {'0', 'o'} or pair == {'1', 'l'}:
            continue
        print("No")
        return
    
    print("Yes")

if __name__ == "__main__":
    main()