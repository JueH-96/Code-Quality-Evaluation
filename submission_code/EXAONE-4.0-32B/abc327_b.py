def main():
    B = int(input().strip())
    for a in range(1, 16):
        power = a ** a
        if power == B:
            print(a)
            return
    print(-1)

if __name__ == '__main__':
    main()