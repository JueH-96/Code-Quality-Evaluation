def main():
    X, Y = map(int, input().split())
    if (Y > X and Y - X <= 2) or (X > Y and X - Y <= 3):
        print("Yes")
    else:
        print("No")

main()