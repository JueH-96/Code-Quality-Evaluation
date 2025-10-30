def main():
    S = input().strip()
    # Extract the numeric part and convert to integer
    num = int(S[3:])
    # It must be between 001 and 349 inclusive, and not 316
    if 1 <= num <= 349 and num != 316:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()