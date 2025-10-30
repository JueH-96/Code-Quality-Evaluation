import sys

def main():
    """
    Main function to solve the interactive problem.
    It determines the most efficient query strategy between two standard methods
    and then executes it to find the sum of A[L...R] mod 100.
    """

    # --- Helper functions for interaction and bit manipulation ---

    def ask(i, j):
        """Sends a query '? i j' to the judge and returns the answer."""
        print(f"? {i} {j}", flush=True)
        response = sys.stdin.readline()
        if not response:
            exit()
        T = int(response)
        if T == -1:
            # Judge indicates an error, implying a bug in our program.
            exit()
        return T

    def answer(S):
        """Submits the final answer '! S' and terminates."""
        print(f"! {S}", flush=True)
        exit()

    def popcount(n):
        """Counts the number of set bits in an integer's binary representation."""
        if n < 0: return 0
        return bin(n).count('1')

    def ctz(n):
        """Counts trailing zeros in an integer's binary representation."""
        if n == 0:
            # ctz(0) is conceptually infinite. A value larger than N_max=18 is sufficient.
            return 60 
        return (n & -n).bit_length() - 1

    # --- Method 1: Greedy decomposition of the interval [L, R] ---
    
    def count_method1(N, L, R):
        """Simulates Method 1 to count the number of queries it would make."""
        count = 0
        current_pos = L
        while current_pos <= R:
            range_len = R - current_pos + 1
            k_len = range_len.bit_length() - 1
            k_align = ctz(current_pos)
            k = min(k_len, k_align)
            
            count += 1
            block_size = 1 << k
            if block_size == 0: break
            current_pos += block_size
        return count

    def solve_method1(N, L, R):
        """Executes Method 1: greedy decomposition of [L, R]."""
        total_sum = 0
        current_pos = L
        while current_pos <= R:
            range_len = R - current_pos + 1
            k_len = range_len.bit_length() - 1
            k_align = ctz(current_pos)
            k = min(k_len, k_align)
            
            i = k
            j = current_pos >> k
            
            res = ask(i, j)
            total_sum = (total_sum + res) % 100
            
            block_size = 1 << k
            current_pos += block_size
            
        answer(total_sum)

    # --- Method 2: S(L,R) = S(0,R) - S(0,L-1) using prefix sums ---

    def count_method2(N, L, R):
        """Calculates the number of queries for Method 2."""
        count = popcount(R + 1)
        if L > 0:
            count += popcount(L)
        return count

    def get_prefix_sum(K, N):
        """Calculates sum(A[0...K]) mod 100 by querying blocks."""
        if K < 0:
            return 0
        
        total = 0
        length_to_cover = K + 1
        current_pos = 0
        
        for i in range(N, -1, -1):
            power_of_2 = 1 << i
            if length_to_cover >= power_of_2:
                j = current_pos >> i
                res = ask(i, j)
                total = (total + res) % 100
                
                current_pos += power_of_2
                length_to_cover -= power_of_2
                
                if length_to_cover == 0:
                    break
        return total

    def solve_method2(N, L, R):
        """Executes Method 2: using prefix sum differences."""
        sum_R = get_prefix_sum(R, N)
        sum_L_minus_1 = get_prefix_sum(L - 1, N)
        
        final_sum = (sum_R - sum_L_minus_1 + 100) % 100
        answer(final_sum)

    # --- Main Logic ---
    
    try:
        N, L, R = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        exit()

    if L == R:
        res = ask(0, L)
        answer(res)
        return

    num_queries1 = count_method1(N, L, R)
    num_queries2 = count_method2(N, L, R)

    if num_queries1 <= num_queries2:
        solve_method1(N, L, R)
    else:
        solve_method2(N, L, R)

if __name__ == "__main__":
    main()