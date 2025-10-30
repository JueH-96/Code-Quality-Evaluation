def main():
    import sys
    input = sys.stdin.readline
    
    s = input().strip()
    
    if s[0].isupper() and (len(s) == 1 or s[1:].islower()):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()