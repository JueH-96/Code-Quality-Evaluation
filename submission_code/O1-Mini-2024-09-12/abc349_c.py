def is_subsequence(s, t):
    it = iter(s)
    return all(c in it for c in t)

def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    option1 = is_subsequence(S, T.lower())
    option2 = False
    if T.endswith('X'):
        prefix = T[:2].lower()
        option2 = is_subsequence(S, prefix)
    
    if option1 or option2:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()