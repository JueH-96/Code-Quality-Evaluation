def main():
    import sys
    input = sys.stdin.readline
    
    S = input().strip()
    # S is in the form ABCxxx, where xxx are digits.
    if len(S) != 6 or not S.startswith("ABC"):
        print("No")
        return
    
    try:
        num = int(S[3:])
    except:
        print("No")
        return
        
    # It is valid if:
    #   S is ABC001 ~ ABC315, or
    #   S is ABC317 ~ ABC349, or
    #   S is ABC... but note: ABC316 is not allowed, and ABC350+ are not allowed.
    # Let's simply check allowed ranges:
    if 1 <= num <= 315:
        print("Yes")
    elif 317 <= num <= 349:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()