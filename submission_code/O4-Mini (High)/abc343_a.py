def main():
    A, B = map(int, input().split())
    s = A + B
    # choose any integer in 0..9 that's not s
    for x in range(10):
        if x != s:
            print(x)
            return

if __name__ == "__main__":
    main()