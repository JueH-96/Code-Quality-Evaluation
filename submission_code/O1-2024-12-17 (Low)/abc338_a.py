def main():
    S = input().strip()
    # Check the condition: first character is uppercase and rest are lowercase
    if S[0].isupper() and S[1:].islower():
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()