def is_subsequence(target, s):
    s_ptr = 0
    for c in target:
        while s_ptr < len(s) and s[s_ptr] != c:
            s_ptr += 1
        if s_ptr >= len(s):
            return False
        s_ptr += 1
    return True

def main():
    import sys
    input = sys.stdin.read().split()
    s = input[0].strip()
    t = input[1].strip()
    
    if len(t) != 3:
        print("No")
        return
    
    # Case 1: Check if T is a 3-letter subsequence
    t_case1 = t.lower()
    if is_subsequence(t_case1, s):
        print("Yes")
        return
    
    # Case 2: Check if T ends with 'X' and the first two form a subsequence
    if t[2] == 'X':
        t_case2 = t[:2].lower()
        if is_subsequence(t_case2, s):
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()