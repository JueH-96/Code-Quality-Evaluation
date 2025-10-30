def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read the integer N from input
    N = int(data[0])
    
    # Pi truncated to 100 decimal places (including the "3." at start)
    pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    
    # We want to output "3." plus the first N digits after the decimal point.
    # The first two characters "3." are at indices 0 and 1, then the digits start at index 2.
    truncated_pi = pi_str[:2 + N]
    
    sys.stdout.write(truncated_pi)

if __name__ == "__main__":
    main()