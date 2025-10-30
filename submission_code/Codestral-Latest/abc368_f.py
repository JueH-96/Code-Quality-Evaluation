import sys
from math import gcd
from functools import reduce

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    # Function to find the greatest common divisor (GCD) of a list of numbers
    def find_gcd(lst):
        return reduce(gcd, lst)

    # Check if all numbers in the list are powers of 2
    def all_powers_of_two(lst):
        return all((x & (x - 1)) == 0 for x in lst)

    # Calculate the GCD of the list
    gcd_value = find_gcd(A)

    # If the GCD is greater than 1, Anna can always win by reducing the GCD
    if gcd_value > 1:
        print("Anna")
    else:
        # If all numbers are powers of 2, Bruno wins
        if all_powers_of_two(A):
            print("Bruno")
        else:
            print("Anna")

if __name__ == "__main__":
    main()