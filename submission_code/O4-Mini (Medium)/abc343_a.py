def main():
    A, B = map(int, input().split())
    s = A + B
    # Pick any number in 0..9 that's not equal to s.
    for i in range(10):
        if i != s:
            print(i)
            return

if __name__ == "__main__":
    main()