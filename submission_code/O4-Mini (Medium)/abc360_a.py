def main():
    S = input().strip()
    # Check positions of 'R' and 'M' in the string
    if S.index('R') < S.index('M'):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()