def main():
    A, B = map(int, input().split())
    s = A + B
    for i in range(10):
        if i != s:
            print(i)
            return

if __name__ == "__main__":
    main()