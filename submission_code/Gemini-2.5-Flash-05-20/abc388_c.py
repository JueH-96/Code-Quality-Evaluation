import sys
from bisect import bisect_right

def solve():
    # Read N, the number of mochi
    N = int(sys.stdin.readline())
    
    # Read the mochi sizes into a list A.
    # The input guarantees that A_i are already sorted in ascending order.
    A = list(map(int, sys.stdin.readline().split()))

    total_kagamimochi = 0

    # Iterate through each mochi A[j] to consider it as the bottom mochi (B)
    for j in range(N):
        current_bottom_mochi_size = A[j]
        
        # Calculate the maximum allowed size for the top mochi (A)
        # The condition for placing mochi A on top of mochi B is: size_A <= size_B / 2
        # So, the maximum size for a valid top mochi is current_bottom_mochi_size / 2.0.
        # We use 2.0 for float division to handle cases where current_bottom_mochi_size is odd
        # (e.g., 7 / 2.0 = 3.5).
        max_top_size = current_bottom_mochi_size / 2.0

        # Find the number of mochi A[i] (potential top mochi) that satisfy A[i] <= max_top_size.
        # Since the list A is sorted, we can use `bisect_right` for efficient counting.
        # `bisect_right(A, x)` returns an insertion point that maintains sorted order.
        # All elements to the left of this insertion point are less than or equal to `x`.
        # Therefore, the returned index is exactly the count of elements `<= x`.
        # This correctly counts distinct mochi by their index, as `A` is a list of distinct elements
        # (even if their values are duplicates, like `A[2]` and `A[3]` in `[2, 3, 4, 4, 7, 10]`).
        count_possible_top_mochi = bisect_right(A, max_top_size)
        
        # An important consideration is whether the top mochi and bottom mochi can be the same.
        # The problem states "choose two mochi ... and place one on top of the other",
        # implying distinct mochi (by index).
        # If A[i] were placed on A[i], the condition A[i] <= A[i] / 2 would have to hold.
        # For any A[i] >= 1 (as per constraints), A[i] / 2.0 will always be strictly less than A[i].
        # For example, if A[i]=1, 1 <= 0.5 is false. If A[i]=2, 2 <= 1.0 is false.
        # Thus, A[j] itself can never be a valid top mochi for A[j].
        # Therefore, we don't need to adjust `count_possible_top_mochi` to exclude `A[j]`.
        
        total_kagamimochi += count_possible_top_mochi
    
    # Print the total count of different kinds of kagamimochi that can be made.
    print(total_kagamimochi)

# Call the solve function to execute the program
solve()