def main():
    S = input().strip()
    # Check if the first character is uppercase
    if not S[0].isupper():
        print("No")
        return
    # Check if all other characters are lowercase
    if all(c.islower() for c in S[1:]):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()