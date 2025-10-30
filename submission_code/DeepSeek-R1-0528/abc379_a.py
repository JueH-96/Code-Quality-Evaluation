def main():
    s = input().strip()
    a, b, c = s[0], s[1], s[2]
    first_num = int(b + c + a)
    second_num = int(c + a + b)
    print(f"{first_num} {second_num}")

if __name__ == "__main__":
    main()