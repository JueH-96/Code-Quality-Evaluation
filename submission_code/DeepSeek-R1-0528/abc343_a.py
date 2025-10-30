def main():
    A, B = map(int, input().split())
    s = A + B
    if s == 9:
        print(8)
    else:
        print(9)

if __name__ == '__main__':
    main()