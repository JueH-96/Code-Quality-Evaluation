def main():
    S = input().strip()
    # Check if the rice plate ('R') is to the left of the miso soup plate ('M')
    if S.index('R') < S.index('M'):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()