def main():
    S = input().strip()
    T = input().strip()
    
    i = len(S) - 1
    j = len(T) - 1
    res = []
    
    while i >= 0 and j >= 0:
        if S[i] == T[j]:
            res.append(j + 1)
            i -= 1
            j -= 1
        else:
            j -= 1
            
    res.reverse()
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()