def main():
    A, B = map(int, input().split())
    # If the two excluded suspects are different, the culprit is the remaining one.
    if A != B:
        # Since 1+2+3 = 6, the culprit is 6 - A - B
        print(6 - A - B)
    else:
        # If A == B, two suspects remain, cannot uniquely identify.
        print(-1)

if __name__ == "__main__":
    main()