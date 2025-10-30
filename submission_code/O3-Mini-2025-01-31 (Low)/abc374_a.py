def main():
    import sys
    input = sys.stdin.readline
    # Read the input string and strip newline
    S = input().strip()
    # Check if S ends with 'san'
    if S.endswith("san"):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()