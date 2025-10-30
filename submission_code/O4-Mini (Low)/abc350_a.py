def main():
    S = input().strip()
    # Extract the numeric part of the string.
    num = int(S[3:])
    # Check if it falls in the ranges [1, 315] or [317, 349].
    if (1 <= num <= 315) or (317 <= num <= 349):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()