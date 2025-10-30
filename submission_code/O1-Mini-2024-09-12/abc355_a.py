# YOUR CODE HERE
def main():
    A, B = map(int, input().split())
    if A != B:
        culprit = 6 - A - B
        print(culprit)
    else:
        print(-1)

if __name__ == "__main__":
    main()