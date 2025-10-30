def main():
    import sys
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()
    t_lower = t.lower()
    
    def is_subsequence(target, source):
        iter_source = iter(source)
        return all(c in iter_source for c in target)
    
    # Check case 1: 3-letter subsequence
    if is_subsequence(t_lower, s):
        print("Yes")
        return
    
    # Check case 2: 2-letter subsequence + X
    if t[2] == 'X':
        two_chars = t_lower[:2]
        if is_subsequence(two_chars, s):
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == "__main__":
    main()