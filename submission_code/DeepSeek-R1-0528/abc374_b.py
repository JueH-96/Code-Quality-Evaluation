def main():
    S = input().strip()
    T = input().strip()
    
    min_len = min(len(S), len(T))
    for i in range(min_len):
        if S[i] != T[i]:
            print(i+1)
            return
            
    if len(S) == len(T):
        print(0)
    else:
        print(min_len + 1)

if __name__ == "__main__":
    main()