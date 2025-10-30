def main():
    S = input().strip()
    upper_count = 0
    lower_count = 0
    for char in S:
        if char.isupper():
            upper_count += 1
        elif char.islower():
            lower_count += 1
    if upper_count > lower_count:
        print(S.upper())
    else:
        print(S.lower())

if __name__ == "__main__":
    main()