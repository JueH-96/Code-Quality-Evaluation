def main():
    B = int(input().strip())
    found = -1
    for A in range(1, 17):
        power = A ** A
        if power == B:
            found = A
            break
        if power > B:
            break
    print(found)

if __name__ == "__main__":
    main()