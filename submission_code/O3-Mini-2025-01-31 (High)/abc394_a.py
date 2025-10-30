def main():
    s = input().strip()
    # Filter only the characters '2' from the string
    result = ''.join(char for char in s if char == '2')
    print(result)

if __name__ == "__main__":
    main()