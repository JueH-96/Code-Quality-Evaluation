def main():
    S = input().strip()
    # Check if the index of 'R' is less than the index of 'M'
    if S.index('R') < S.index('M'):
        print("Yes")
    else:
        print("No")

# Do not forget to call main
main()