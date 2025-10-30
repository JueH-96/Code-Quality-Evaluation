def main():
    S = input().strip()
    if S.index('R') < S.index('M'):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()