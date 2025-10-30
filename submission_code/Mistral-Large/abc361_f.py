import sys
import math

def count_powers(N):
    count = 0
    seen = set()

    # Check for powers of 2
    power_of_2 = 1
    while power_of_2 <= N:
        seen.add(power_of_2)
        power_of_2 *= 2

    # Check for powers of other integers
    for a in range(3, int(math.isqrt(N)) + 1):
        power = a * a
        while power <= N:
            seen.add(power)
            power *= a

    return len(seen)

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    result = count_powers(N)
    print(result)

if __name__ == "__main__":
    main()