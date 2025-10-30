import math

def count_good_integers(N):
    count = 0
    power_of_2 = 2  # Start with 2^1
    
    while power_of_2 <= N:
        # For each power of 2, count how many odd perfect squares we can multiply it by
        # while staying under N
        limit = N // power_of_2
        
        # Number of odd perfect squares up to limit
        # Perfect squares up to limit: 1, 4, 9, 16, 25, ...
        # Odd perfect squares: 1, 9, 25, ... (approximately half of all perfect squares)
        num_odd_squares = (math.isqrt(limit) + 1) // 2
        
        count += num_odd_squares
        power_of_2 *= 2
    
    return count

if __name__ == "__main__":
    N = int(input())
    print(count_good_integers(N))