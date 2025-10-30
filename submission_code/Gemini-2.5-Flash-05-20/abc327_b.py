import sys

def solve():
    B = int(sys.stdin.readline())

    found_A = -1
    
    # We are looking for a positive integer A such that A^A = B.
    # Let's analyze the growth of A^A:
    # 1^1 = 1
    # 2^2 = 4
    # 3^3 = 27
    # 4^4 = 256
    # 5^5 = 3,125
    # 6^6 = 46,656
    # 7^7 = 823,543
    # 8^8 = 16,777,216
    # 9^9 = 387,420,489
    # 10^10 = 10,000,000,000 (10^10)
    # 11^11 = 285,311,670,611 (approx 2.8 * 10^11)
    # 12^12 = 8,916,100,448,256 (approx 8.9 * 10^12)
    # 13^13 = 302,875,106,592,253 (approx 3 * 10^14)
    # 14^14 = 11,112,006,825,558,016 (approx 1.1 * 10^16)
    # 15^15 = 437,893,890,380,859,375 (approx 4.3 * 10^17)
    # 16^16 = 18,446,744,073,709,551,616 (approx 1.8 * 10^19)

    # The maximum value for B is 10^18.
    # As A^A grows very rapidly, we can see that if A is 16, A^A is already
    # larger than 10^18. This means that if a solution A exists, it must be
    # less than or equal to 15.
    # Therefore, we only need to check A values from 1 up to 15.

    for A in range(1, 16): # A will take values 1, 2, ..., 15
        current_A_pow_A = A ** A
        
        if current_A_pow_A == B:
            found_A = A
            break
        
        # If A^A is already greater than B, then for any larger A' (A' > A),
        # A'^A' will also be greater than B (since A^A is a strictly increasing
        # function for A >= 1). So, we can stop searching.
        if current_A_pow_A > B:
            break
    
    sys.stdout.write(str(found_A) + "
")

solve()