def main():
    S = input().strip()
    # Check positions of 'R' (rice) and 'M' (miso soup)
    if S.index('R') < S.index('M'):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()