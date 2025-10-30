def main():
    import sys
    data = sys.stdin.read().split()
    # Parse the six integers from input
    x_A, y_A, x_B, y_B, x_C, y_C = map(int, data)
    
    # Calculate squared distances between points
    d_AB = (x_B - x_A) ** 2 + (y_B - y_A) ** 2
    d_BC = (x_C - x_B) ** 2 + (y_C - y_B) ** 2
    d_CA = (x_A - x_C) ** 2 + (y_A - y_C) ** 2
    
    # Check for the Pythagorean condition for a right triangle
    if d_AB + d_BC == d_CA or d_AB + d_CA == d_BC or d_BC + d_CA == d_AB:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

# Call main to execute the solution function
if __name__ == '__main__':
    main()