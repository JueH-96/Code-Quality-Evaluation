def main():
    # Store the integer part and the 100-digit fractional part of pi as strings.
    pi_integer_part = "3"
    pi_fractional_part = (
        "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    )
    
    import sys
    N = int(sys.stdin.read().strip())
    
    # Truncate (slice) the fractional part up to N digits, then combine with the integer part.
    result = pi_integer_part + "." + pi_fractional_part[:N]
    print(result)

# Do not remove this call to main
main()