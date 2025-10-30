def main():
    X, Y = map(int, input().split())
    diff = Y - X
    # Moving up
    if diff > 0:
        if diff <= 2:
            print("Yes")
        else:
            print("No")
    # Moving down
    else:
        if -diff <= 3:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()