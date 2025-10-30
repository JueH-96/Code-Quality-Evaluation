# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    # Count the number of days he needs a logo T-shirt
    logo_days = S.count('2')
    
    # Calculate the number of logo T-shirts needed
    logo_tshirts_needed = max(0, logo_days - M)
    
    # Print the result
    print(logo_tshirts_needed)

if __name__ == "__main__":
    main()