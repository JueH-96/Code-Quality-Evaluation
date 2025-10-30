def main():
    import sys
    N = sys.stdin.read().strip()  # read the input as a string
    
    # Check if the digit counts satisfy the given conditions:
    # - '1' appears exactly once.
    # - '2' appears exactly twice.
    # - '3' appears exactly three times.
    if N.count('1') == 1 and N.count('2') == 2 and N.count('3') == 3:
        print("Yes")
    else:
        print("No")
        
# Do not forget to call main
if __name__ == '__main__':
    main()