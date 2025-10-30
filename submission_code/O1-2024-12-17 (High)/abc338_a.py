def main():
    S = input().strip()
    # Check if the first character is uppercase
    # and all the remaining characters (if any) are lowercase
    if S[0].isupper() and all(c.islower() for c in S[1:]):
        print("Yes")
    else:
        print("No")

main()