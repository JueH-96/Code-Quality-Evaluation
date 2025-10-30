def main():
    n = int(input().strip())
    s = input().strip()
    for i in range(len(s) - 1):
        if (s[i] == 'a' and s[i+1] == 'b') or (s[i] == 'b' and s[i+1] == 'a'):
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()