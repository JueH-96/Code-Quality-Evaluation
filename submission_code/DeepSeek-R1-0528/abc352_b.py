def main():
    S = input().strip()
    T = input().strip()
    
    res = []
    s_index = 0
    n = len(S)
    
    for pos, char in enumerate(T, start=1):
        if s_index < n and char == S[s_index]:
            res.append(pos)
            s_index += 1
            
    print(" ".join(map(str, res)))

if __name__ == '__main__':
    main()