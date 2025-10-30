def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    X = int(data[0])
    Y = int(data[1])
    
    # If moving up, Takahashi uses stairs if it's 2 floors or less.
    if Y > X:
        if Y - X <= 2:
            print("Yes")
        else:
            print("No")
    # If moving down, he uses the stairs if it's 3 floors or less.
    else:  # Y < X because X != Y per constraints.
        if X - Y <= 3:
            print("Yes")
        else:
            print("No")
            
if __name__ == '__main__':
    main()