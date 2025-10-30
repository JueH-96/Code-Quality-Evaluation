def main():
    import sys
    X, Y = map(int, sys.stdin.read().split())
    
    if Y > X:
        # Going up; use stairs if the difference is 2 floors or less
        if Y - X <= 2:
            print("Yes")
        else:
            print("No")
    else:
        # Going down; use stairs if the difference is 3 floors or less
        if X - Y <= 3:
            print("Yes")
        else:
            print("No")
            
if __name__ == '__main__':
    main()