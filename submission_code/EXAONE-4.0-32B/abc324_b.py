def main():
    n = int(input().strip())
    temp = n
    while temp % 2 == 0:
        temp //= 2
    while temp % 3 == 0:
        temp //= 3
    print("Yes" if temp == 1 else "No")

if __name__ == "__main__":
    main()