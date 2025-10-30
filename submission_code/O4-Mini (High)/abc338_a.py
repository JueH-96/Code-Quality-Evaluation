def main():
    S = input().strip()
    # Check if first character is uppercase
    if not S[0].isupper():
        print("No")
        return
    # Check if all other characters are lowercase
    for c in S[1:]:
        if not c.islower():
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()