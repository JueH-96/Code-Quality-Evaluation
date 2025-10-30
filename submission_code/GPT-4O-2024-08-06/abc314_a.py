# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # The value of pi to 100 decimal places
    pi_100 = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    
    # Truncate pi to N decimal places
    result = pi_100[:2 + N]  # 2 because "3." is the first two characters
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()