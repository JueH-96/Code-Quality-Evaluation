def solve():
    import sys
    
    # Read the integer N
    N = int(sys.stdin.readline().strip())
    
    # Pi to 100 decimal places as given in the problem statement
    pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    
    # Truncate pi to N decimal places (the substring includes the "3.")
    result = pi_str[:N+2]
    
    # Print the result
    print(result)

def main():
    solve()

if __name__ == "__main__":
    main()