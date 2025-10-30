def main():
    import sys
    input_data = sys.stdin.read().split()
    L, R = map(int, input_data)
    
    if L == 1 and R == 0:
        print("Yes")
    elif L == 0 and R == 1:
        print("No")
    else:
        print("Invalid")

if __name__ == '__main__':
    main()