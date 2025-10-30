def main():
    s = input().strip()
    # Check that each digit is strictly greater than the next
    for i in range(len(s) - 1):
        if s[i] <= s[i + 1]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()