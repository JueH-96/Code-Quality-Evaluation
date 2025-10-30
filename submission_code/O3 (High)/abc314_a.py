import sys

def main():
    # π to 100 decimal places (the dot counts as one character in the string)
    pi_100 = (
        "3.1415926535897932384626433832795028841971693993751058209749445923"
        "078164062862089986280348253421170679"
    )
    
    # Read N
    n_line = sys.stdin.readline().strip()
    if not n_line:
        return
    N = int(n_line)
    
    # Truncate to N decimal places: keep '3.' (2 chars) + N digits
    result = pi_100[:2 + N]
    
    # Output
    print(result)

if __name__ == "__main__":
    main()