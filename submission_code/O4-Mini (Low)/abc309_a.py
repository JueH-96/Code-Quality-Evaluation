def main():
    A, B = map(int, input().split())
    # They are adjacent horizontally if B = A + 1
    # and A is not at the right edge of a row (i.e., A % 3 != 0).
    if B - A == 1 and A % 3 != 0:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()