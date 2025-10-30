def main():
    S = input().strip()
    T = input().strip()
    
    def is_sub(s, pat):
        idx = 0
        for char in s:
            if idx < len(pat) and char == pat[idx]:
                idx += 1
        return idx == len(pat)
    
    pattern1 = T.lower()
    if is_sub(S, pattern1):
        print("Yes")
    elif T[2] == 'X':
        pattern2 = T[:2].lower()
        if is_sub(S, pattern2):
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == "__main__":
    main()