def main():
    import sys
    S = sys.stdin.readline().strip()
    # Check if first character is uppercase
    # and all remaining characters are lowercase
    if S and S[0].isupper() and all(c.islower() for c in S[1:]):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()