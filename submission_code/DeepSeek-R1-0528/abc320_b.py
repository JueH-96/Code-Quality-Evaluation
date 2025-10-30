def main():
    S = input().strip()
    n = len(S)
    if n == 0:
        print(0)
        return
        
    max_len = 1
    for i in range(n):
        l, r = i, i
        while l >= 0 and r < n and S[l] == S[r]:
            if r - l + 1 > max_len:
                max_len = r - l + 1
            l -= 1
            r += 1
        
        l, r = i, i + 1
        while l >= 0 and r < n and S[l] == S[r]:
            if r - l + 1 > max_len:
                max_len = r - l + 1
            l -= 1
            r += 1
            
    print(max_len)

if __name__ == "__main__":
    main()