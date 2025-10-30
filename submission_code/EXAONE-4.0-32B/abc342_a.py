def main():
    s = input().strip()
    if s[0] == s[1]:
        majority = s[0]
        for i in range(2, len(s)):
            if s[i] != majority:
                print(i + 1)
                return
    else:
        if s[0] == s[2]:
            print(2)
        else:
            print(1)

if __name__ == "__main__":
    main()