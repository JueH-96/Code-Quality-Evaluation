def main():
    # Pi up to 100 decimal places (total length = 2 + 100 = 102 characters)
    pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    
    # Read input
    import sys
    N = int(sys.stdin.readline().strip())
    
    # Truncate pi to N decimal places
    # "3." is the first 2 characters, so we slice up to 2+N
    result = pi_str[:2 + N]
    
    # Print the result
    print(result)

# Call the main function
if __name__ == "__main__":
    main()