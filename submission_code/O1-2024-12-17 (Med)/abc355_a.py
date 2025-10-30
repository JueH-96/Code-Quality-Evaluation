def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)

    suspects = {1, 2, 3}
    suspects.discard(A)
    suspects.discard(B)

    if len(suspects) == 1:
        print(suspects.pop())
    else:
        print(-1)

if __name__ == "__main__":
    main()