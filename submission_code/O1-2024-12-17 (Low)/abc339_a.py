def main():
    S = input().strip()
    # Split the string by '.' and print the last part.
    result = S.split('.')[-1]
    print(result)

if __name__ == "__main__":
    main()