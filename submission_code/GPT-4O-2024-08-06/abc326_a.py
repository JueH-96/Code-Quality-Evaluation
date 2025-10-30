# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    X, Y = map(int, data.split())
    
    # Calculate the difference
    difference = abs(X - Y)
    
    # Determine if Takahashi uses the stairs or the elevator
    if difference <= 2 or (X > Y and difference <= 3):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()