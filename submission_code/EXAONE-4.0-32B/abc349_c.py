def main():
    S = input().strip()
    T = input().strip()
    
    def is_subsequence(pattern, text):
        if not pattern:
            return True
        idx = 0
        for char in text:
            if idx < len(pattern) and char == pattern[idx]:
                idx += 1
            if idx == len(pattern):
                break
        return idx == len(pattern)
    
    t_lower = T.lower()
    if is_subsequence(t_lower, S):
        print("Yes")
    elif T[2] == 'X' and is_subsequence(t_lower[:2], S):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()