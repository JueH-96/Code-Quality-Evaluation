def main():
    S = input().strip()
    upper_count = sum(1 for char in S if char.isupper())
    lower_count = sum(1 for char in S if char.islower())
    if upper_count > lower_count:
        print(S.upper())
    else:
        print(S.lower())

if __name__ == "__main__":
    main()