def main():
    S = input().strip()
    # Extract the numeric part
    num = int(S[3:])
    # Check if it's in the valid ranges and not 316
    if 1 <= num <= 349 and num != 316:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()