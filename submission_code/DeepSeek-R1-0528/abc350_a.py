def main():
    s = input().strip()
    num_str = s[3:]
    num = int(num_str)
    if 1 <= num <= 349 and num != 316:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()