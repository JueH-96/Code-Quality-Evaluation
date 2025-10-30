def main():
    count = 0
    # We know there are 12 strings provided
    for i in range(1, 13):
        s = input().strip()
        if len(s) == i:
            count += 1
    print(count)

if __name__ == '__main__':
    main()