import sys
import math

def main():
    input = sys.stdin.read
    N = int(input().strip())
    
    # Set to store unique powers
    powers = set()
    
    # We start with a = 2 since a^b for a=1 is always 1 and not considered here
    a = 2
    while True:
        b = 2
        while True:
            power = a ** b
            if power > N:
                break
            powers.add(power)
            b += 1
        if a ** 2 > N:
            break
        a += 1
    
    # Output the number of unique powers
    print(len(powers))

if __name__ == "__main__":
    main()