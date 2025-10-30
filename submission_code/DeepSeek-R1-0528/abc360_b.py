def main():
    S, T = input().split()
    n = len(S)
    
    for w in range(1, n):
        chunks = [S[i:i+w] for i in range(0, n, w)]
        for c in range(1, w + 1):
            res_str = ""
            for chunk in chunks:
                if len(chunk) >= c:
                    res_str += chunk[c - 1]
            if res_str == T:
                print("Yes")
                return
                
    print("No")

if __name__ == "__main__":
    main()