def main():
    B = int(input().strip())
    A = 1
    while True:
        power = A ** A
        if power == B:
            print(A)
            return
        elif power > B:
            print(-1)
            return
        A += 1

if __name__ == "__main__":
    main()