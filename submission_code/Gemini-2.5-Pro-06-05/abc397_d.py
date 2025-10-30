import math
import sys

def main():
    """
    Main function to solve the problem.
    """
    try:
        N_str = sys.stdin.readline()
        if not N_str:
            return
        N = int(N_str)
    except (IOError, ValueError):
        return

    # The equation is x^3 - y^3 = N, with x, y being positive integers.
    # This implies x > y. Let x = y + a, where a is a positive integer.
    # N = (y+a)^3 - y^3 = 3ay^2 + 3a^2y + a^3
    # N = a * (3y^2 + 3ay + a^2)
    # This means a must be a factor of N. Let b = N/a.
    # b = 3y^2 + 3ay + a^2
    #
    # Rearranging to solve for y:
    # 4b - a^2 = 12y^2 + 12ay + 3a^2 = 3(4y^2 + 4ay + a^2) = 3(2y+a)^2
    # So we need to check if (4b - a^2) is divisible by 3 and the result is a perfect square.
    # Let Z^2 = (4b - a^2)/3. Then Z = 2y+a.
    # From this, y = (Z-a)/2. For y to be a positive integer, Z-a must be positive and even.
    #
    # We also have an upper bound on a:
    # 4b >= a^2 => 4(N/a) >= a^2 => 4N >= a^3 => a <= (4N)^(1/3).
    # For N <= 10^18, a <= (4*10^18)^(1/3) ~= 1.587 * 10^6.
    # We iterate 'a' up to a safe limit.

    limit = 2 * 10**6
    found = False
    for a in range(1, limit + 1):
        if N % a == 0:
            b = N // a
            
            term = 4 * b - a * a

            if term < 0:
                continue

            if term % 3 != 0:
                continue

            Z_squared = term // 3
            
            if Z_squared > 0:
                # math.isqrt is an efficient way to find the integer square root.
                Z = math.isqrt(Z_squared)

                if Z * Z == Z_squared:
                    # Conditions for y to be a positive integer:
                    # 1. y > 0 => (Z - a) / 2 > 0 => Z > a
                    # 2. y is integer => (Z - a) is even
                    if (Z > a) and ((Z - a) % 2 == 0):
                        y = (Z - a) // 2
                        x = y + a
                        
                        # A valid pair is found.
                        print(x, y)
                        found = True
                        break
    
    if not found:
        print(-1)

if __name__ == "__main__":
    main()