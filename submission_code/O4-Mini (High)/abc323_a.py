def main():
    s = input().strip()
    # Check even positions (2,4,...,16) i.e. indices 1,3,...,15 in 0-based indexing
    for i in range(1, 16, 2):
        if s[i] != '0':
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()