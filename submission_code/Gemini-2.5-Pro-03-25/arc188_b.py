# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    # Read N and K for the current test case from standard input
    # N: number of equally spaced points on the circle (numbered 0 to N-1)
    # K: Bob's starting position (Alice starts at point 0)
    N, K = map(int, sys.stdin.readline().split())
    
    # The problem asks whether it's possible for Alice and Bob, cooperating,
    # to color all N points black. They take turns coloring a white point black.
    # The crucial rule is that after each move, the coloring must be symmetric
    # with respect to the line connecting the current player's position and the center of the circle.
    
    # Let's analyze the symmetry condition and its implications.
    # Alice is at point 0. The reflection operation with respect to her axis is Ref_0(i) = (-i) % N.
    # Bob is at point K. The reflection operation with respect to his axis is Ref_K(i) = (2*K - i) % N.
    
    # The game dynamics change significantly if Alice's and Bob's symmetry axes are identical.
    # This occurs if Ref_0(i) == Ref_K(i) for all i.
    # (-i) % N == (2*K - i) % N
    # This equality holds if and only if 0 == 2*K % N, which means N must divide 2*K.
    
    # Case 1: N is odd.
    # If N is odd, gcd(N, 2) = 1. So N | 2K implies N | K.
    # However, the problem states 1 <= K <= N-1. K cannot be a multiple of N.
    # Thus, if N is odd, N never divides 2K, meaning Ref_0 is always different from Ref_K.
    
    # Case 2: N is even.
    # If N is even, let N = 2m. N | 2K means 2m | 2K, which simplifies to m | K.
    # This means K must be a multiple of m = N/2.
    # Since 1 <= K <= N-1, the only possibility for K to be a multiple of N/2 is K = N/2.
    
    # Conclusion: The reflection operations Ref_0 and Ref_K are identical if and only if N is even and K = N/2.
    
    # What happens if Ref_0 == Ref_K? (i.e., N is even and K = N/2)
    # In this case, the state of the board must always remain symmetric with respect to this common axis.
    # A detailed analysis of the game rules reveals that a player can only validly color a point x if:
    # 1. x is self-reflective with respect to the player's axis (i.e., x = Ref_P(x)).
    # 2. x is not self-reflective, but its reflection y = Ref_P(x) is already black.
    # When Ref_0 = Ref_K, the game state must always be symmetric. If a state B is symmetric and x is a white point,
    # its reflection y = Ref_P(x) must also be white (unless x=y).
    # This implies condition 2 can never be met starting from a symmetric state required in this case.
    # Therefore, players can only choose self-reflective points to color.
    
    # The self-reflective points for the axis through 0 (which is the same as the axis through N/2)
    # are the solutions to 2*i = 2*0 % N, which simplifies to 2*i = 0 % N.
    # Since N is even, the solutions are i=0 and i=N/2.
    
    # So, when N is even and K = N/2, Alice and Bob can collectively color at most the two points 0 and N/2.
    # If N > 2, there are more than two points on the circle. Points other than 0 and N/2 will remain white.
    # In this situation (N even, K=N/2, N>2), not all points can be colored black. The answer is "No".
    
    # There is one edge case: N = 2.
    # If N = 2, the points are 0 and 1. K must be 1. This fits the N even, K = N/2 condition.
    # The self-reflective points are 0 and N/2=1. These are all the points on the circle.
    # Alice can color one point (say, 1). Bob can color the other point (0).
    # All points (0 and 1) get colored black. So for N=2, the answer is "Yes".
    
    # Summarizing the condition for "No":
    # The answer is "No" if and only if (N is even AND K == N // 2 AND N > 2).
    
    # Check if the conditions for the "No" case are met
    is_no_case = False
    if N % 2 == 0:  # Check if N is even
        # Using integer division N // 2
        if K == N // 2: # Check if K is exactly half of N
            # Check if N is greater than 2. This excludes the N=2 case.
            if N > 2:
                is_no_case = True

    # Print the result based on whether the "No" condition is met
    if is_no_case:
        print("No")
    else:
        # In all other scenarios (N odd, or N even but K != N/2, or the special case N=2),
        # it is assumed possible to color all points black.
        print("Yes")

# Read the total number of test cases T
T = int(sys.stdin.readline())
# Process each test case by calling the solve function
for _ in range(T):
    solve()