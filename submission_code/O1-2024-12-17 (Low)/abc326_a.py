def main():
    X, Y = map(int, input().split())
    
    # If moving up and the number of floors to move is 2 or less, use stairs
    if X < Y and (Y - X) <= 2:
        print("Yes")
    # If moving down and the number of floors to move is 3 or less, use stairs
    elif X > Y and (X - Y) <= 3:
        print("Yes")
    # Otherwise, use the elevator
    else:
        print("No")

# Do not forget to call main
main()