import sys
import math

def count_good_integers(N):
    good_integers = set()
    
    # We need to consider powers of 2 up to the maximum that when squared won't exceed N
    max_a = math.isqrt(N)  # This is the maximum value b can take
    max_power_of_2 = N.bit_length()  # This is the maximum power of 2 we need to consider
    
    # Iterate over all possible values of a where 2^a is considered
    for a in range(max_power_of_2 + 1):
        power_of_2 = 2 ** a
        
        # Now iterate over possible values of b
        b = 1
        while True:
            value = power_of_2 * (b ** 2)
            if value > N:
                break
            good_integers.add(value)
            b += 1
    
    return len(good_integers)

def main():
    input = sys.stdin.read
    N = int(input().strip())
    result = count_good_integers(N)
    print(result)

if __name__ == "__main__":
    main()