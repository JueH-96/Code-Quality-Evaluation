def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    N = int(input_data)
    # The given pi value to 100 decimal places
    pi_hundred = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    
    # Since the pi string has 1 digit before the decimal point and 1 decimal point,
    # the decimal digits start at index 2 in the string.
    # Truncate to N decimal places:
    # That means: Take everything from beginning up to index 2+N.
    result = pi_hundred[:2+N]
    
    # Print the result
    sys.stdout.write(result)

if __name__ == "__main__":
    main()