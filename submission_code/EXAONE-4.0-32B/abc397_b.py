def main():
    S = input().strip()
    state = 0
    insertions = 0
    for c in S:
        if state == 0:
            if c == 'i':
                state = 1
            else:
                insertions += 1
                state = 0
        else:
            if c == 'o':
                state = 0
            else:
                insertions += 1
                state = 1
    if state == 1:
        insertions += 1
    print(insertions)

if __name__ == "__main__":
    main()