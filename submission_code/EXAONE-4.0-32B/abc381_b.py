def main():
    S = input().strip()
    n = len(S)
    if n % 2 != 0:
        print("No")
        return
    
    for i in range(0, n, 2):
        if S[i] != S[i+1]:
            print("No")
            return
            
    from collections import Counter
    freq = Counter(S)
    for count in freq.values():
        if count != 2:
            print("No")
            return
            
    print("Yes")

if __name__ == "__main__":
    main()