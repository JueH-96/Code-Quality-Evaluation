def main():
    S, T = input().strip().split()
    
    length_S = len(S)
    length_T = len(T)
    
    found = False
    
    # w ranges from 1 to length_S - 1
    for w in range(1, length_S):
        # c ranges from 1 to w
        for c in range(1, w + 1):
            result = ''
            # Split S into substrings of length w
            for i in range(0, length_S, w):
                substring = S[i:i+w]
                # Check if substring length is at least c
                if len(substring) >= c:
                    # 0-based index
                    result += substring[c-1]
            # Check if the result matches T
            if result == T:
                found = True
                break
        if found:
            break
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()