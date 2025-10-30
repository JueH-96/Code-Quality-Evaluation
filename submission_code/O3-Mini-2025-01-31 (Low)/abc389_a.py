def main():
    s = input().strip()
    # The input is guaranteed to have exactly three characters: digit, 'x', digit.
    a = int(s[0])
    b = int(s[2])
    print(a * b)

if __name__ == '__main__':
    main()