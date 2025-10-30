# YOUR CODE HERE
import sys

def main():
    line = sys.stdin.readline().split()
    A = int(line[0])
    B = int(line[1])

    # We need to find the smallest integer k such that A - k * B <= 0.
    # This is equivalent to k * B >= A, or k >= A / B.
    # So, k is the ceiling of A / B.
    # For positive integers A and B, ceil(A / B) can be calculated as (A + B - 1) // B
    # using integer division.
    
    # Constraints: 1 <= A, B <= 10^18. Python handles large integers.
    # A + B - 1 will not overflow standard 64-bit int if this was C++/Java, 
    # but Python handles it regardless.
    # The result (number of attacks) can also be large, up to 10^18 if B=1.
    
    num_attacks = (A + B - 1) // B
    
    sys.stdout.write(str(num_attacks) + "
")

if __name__ == '__main__':
    main()