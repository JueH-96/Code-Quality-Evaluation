def main():
    s = input().strip()
    # Check if the first character is uppercase and the rest are exactly their lowercase form.
    if s[0].isupper() and s[1:] == s[1:].lower():
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()