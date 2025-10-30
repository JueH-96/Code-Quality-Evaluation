def main():
    import sys

    # The value of pi to 100 decimal places (after the decimal point)
    pi_decimals = (
        "14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679"
    )

    # Read input N
    N = int(sys.stdin.readline().strip())

    # Truncate to N decimal places
    result = "3." + pi_decimals[:N]

    # Print the result
    print(result)

if __name__ == "__main__":
    main()