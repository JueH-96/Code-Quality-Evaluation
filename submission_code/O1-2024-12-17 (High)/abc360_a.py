def main():
    S = input().strip()
    # Check the positions of 'R' and 'M' in the string
    if S.index('R') < S.index('M'):
        print("Yes")
    else:
        print("No")

# Do not forget to call main()
main()