import sys
import math # Not needed for floor division with //

# Function to compute floor division which is default behavior of // in Python 3
# No separate function needed unless compatibility with Python 2 is a concern.
# Python 3's // operator performs floor division correctly for negative numbers.
# e.g., -7 // 3 == -3, which is floor(-7/3) = floor(-2.33...) = -3.

def solve():
    """
    Solves a single test case. Reads input values for K, S_x, S_y, T_x, T_y,
    calculates the region indices (i, j) for start and target points,
    transforms these indices into (I, J) coordinates, and computes the
    Manhattan distance in the (I, J) system, which seems to be the minimum
    number of moves based on problem structure and sample verification.
    """
    K, Sx, Sy, Tx, Ty = map(int, sys.stdin.readline().split())

    # Calculate the region indices (i, j) for start point (Sx, Sy).
    # The point used is (Sx + 0.5, Sy + 0.5). The tile containing this point
    # belongs to region R(i, j) where i = floor((Sx + 0.5) / K) and j = floor((Sy + 0.5) / K).
    # It turns out this is equivalent to i = Sx // K and j = Sy // K for integer Sx, Sy and K >= 2.
    # Python's // operator performs floor division.
    iS = Sx // K
    jS = Sy // K
    
    # Calculate the region indices (i, j) for target point (Tx, Ty).
    iT = Tx // K
    jT = Ty // K

    # Transform the region indices (i, j) into a new coordinate system (I, J)
    # where I = i + j and J = i - j.
    # This transformation rotates the grid by 45 degrees and scales it.
    IS = iS + jS
    JS = iS - jS
    IT = iT + jT
    JT = iT - jT
    
    # Based on analysis and sample verification, the minimum number of moves required
    # appears to be the Manhattan distance between the start and target points
    # in the transformed (I, J) coordinate system.
    # Manhattan distance = |I_S - I_T| + |J_S - J_T|.
    # This formula is equivalent to 2 * max(|iS - iT|, |jS - jT|).
    
    dist = abs(IS - IT) + abs(JS - JT)
    
    # Print the result for the current test case.
    print(dist)

# Read the number of test cases.
T = int(sys.stdin.readline())
# Iterate through each test case and solve it.
for _ in range(T):
    solve()