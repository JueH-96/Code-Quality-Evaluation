def main():
    X, Y = map(int, input().split())
    if Y > X:
        # Going up
        if Y - X <= 2:
            print("Yes")
        else:
            print("No")
    else:
        # Going down
        if X - Y <= 3:
            print("Yes")
        else:
            print("No")

# Do not remove the following call
main()