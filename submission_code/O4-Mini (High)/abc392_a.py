def main():
    # Read three integers from standard input
    a, b, c = map(int, input().split())
    
    # Check all pairwise products
    if a * b == c or a * c == b or b * c == a:
        print("Yes")
    else:
        print("No")

# Call main to execute the solution
main()