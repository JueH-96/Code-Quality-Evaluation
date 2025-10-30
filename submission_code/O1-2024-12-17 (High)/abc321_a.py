def main():
    # Read the input integer N as a string
    N = input().strip()
    
    # Check consecutive digits to confirm strictly decreasing order
    # i.e., the i-th digit > (i+1)-th digit
    for i in range(len(N) - 1):
        if not (N[i] > N[i+1]):
            print("No")
            return
            
    print("Yes")

# Call the main function
main()