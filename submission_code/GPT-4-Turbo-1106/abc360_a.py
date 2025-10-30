# Read the input string from standard input
S = input().strip()

# Check if the index of R is less than the index of M
if S.index('R') < S.index('M'):
    print("Yes")
else:
    print("No")