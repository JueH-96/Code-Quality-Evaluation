def main():
    s = input().strip()
    for i in range(1, len(s)):
        if s[i-1] > s[i]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()