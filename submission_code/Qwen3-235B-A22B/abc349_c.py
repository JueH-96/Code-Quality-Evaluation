def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    def is_subsequence(target, s):
        i = 0
        for c in s:
            if i < len(target) and c == target[i].lower():
                i += 1
            if i == len(target):
                break
        return i == len(target)
    
    ok = is_subsequence(T, S)
    if not ok and T[2] == 'X':
        ok = is_subsequence(T[:2], S)
    
    print("Yes" if ok else "No")

if __name__ == "__main__":
    main()