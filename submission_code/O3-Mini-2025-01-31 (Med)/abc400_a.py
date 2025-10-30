def main():
    # Read input A from standard input and convert it to integer
    A = int(input().strip())
    
    # Check if A divides 400 and calculate B if possible.
    if 400 % A == 0:
        B = 400 // A
        print(B)
    else:
        print(-1)

# Call the main function
if __name__ == "__main__":
    main()