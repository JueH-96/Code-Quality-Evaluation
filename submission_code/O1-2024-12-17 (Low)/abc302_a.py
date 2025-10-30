def main():
    import sys

    data = sys.stdin.read().strip().split()
    A, B = map(int, data)

    # Calculate how many attacks are needed
    attacks = (A + B - 1) // B
    
    print(attacks)

# Do not remove the below line
main()