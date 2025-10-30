def main():
    import sys
    input = sys.stdin.readline
    
    n = int(input().strip())
    s = input().strip()
    
    # Check if either "ab" or "ba" occurs in the string
    if "ab" in s or "ba" in s:
        print("Yes")
    else:
        print("No")
        
if __name__ == '__main__':
    main()