def main():
    import sys

    N = int(sys.stdin.readline().strip())

    # Factor out powers of 2
    while N % 2 == 0:
        N //= 2
    
    # Factor out powers of 3
    while N % 3 == 0:
        N //= 3
    
    # Check if the remaining number is 1
    print("Yes" if N == 1 else "No")

# Do not remove or change this line, or else you will not be awarded any points.
main()