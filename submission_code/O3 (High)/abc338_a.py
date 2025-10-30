def main():
    import sys

    S = sys.stdin.readline().rstrip()

    # Check if the first character is uppercase
    first_is_upper = S[0].isupper()

    # Check if all remaining characters (if any) are lowercase
    rest_are_lower = all(c.islower() for c in S[1:])

    if first_is_upper and rest_are_lower:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()