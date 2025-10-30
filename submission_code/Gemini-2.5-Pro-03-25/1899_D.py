# YOUR CODE HERE
# -*- coding: utf-8 -*-
import sys
from collections import Counter

def solve():
    """
    Solves a single test case. Reads input N and array A,
    computes the number of pairs (i, j) with i < j such that
    b_i^(b_j) = b_j^(b_i), where b_k = 2^(a_k).

    The condition b_i^(b_j) = b_j^(b_i) can be rewritten using the definitions:
    (2^(a_i))^(2^(a_j)) = (2^(a_j))^(2^(a_i))
    2^(a_i * 2^(a_j)) = 2^(a_j * 2^(a_i))

    Since the base is 2 (which is > 1), the equality holds if and only if the exponents are equal:
    a_i * 2^(a_j) = a_j * 2^(a_i)

    This equality can be rearranged (assuming a_i, a_j > 0, which is given) to:
    a_i / 2^(a_i) = a_j / 2^(a_j)

    We need to find the number of pairs (i, j) with i < j such that this equality holds.
    Let's analyze the function f(x) = x / 2^x for positive integers x.
    f(1) = 1 / 2^1 = 1/2
    f(2) = 2 / 2^2 = 2/4 = 1/2
    f(3) = 3 / 2^3 = 3/8
    f(4) = 4 / 2^4 = 4/16 = 1/4
    
    For x >= 2, the function f(x) is strictly decreasing:
    f(x) > f(x+1)  <=>  x / 2^x > (x+1) / 2^(x+1)
    Multiply by 2^(x+1): x * 2 > x+1  <=>  2x > x+1  <=> x > 1.
    This inequality holds for all integers x >= 2.
    
    Therefore, f(x) = f(y) for distinct positive integers x, y holds if and only if {x, y} = {1, 2}.
    
    So the condition a_i / 2^(a_i) = a_j / 2^(a_j) holds if and only if:
    1. a_i = a_j
    2. {a_i, a_j} = {1, 2}

    The function counts pairs satisfying these conditions by summing up
    counts from these two disjoint cases:
    Case 1: Pairs (i, j) with i < j where a_i = a_j.
    Case 2: Pairs (i, j) with i < j where {a_i, a_j} = {1, 2}.
    """
    
    # Read N, the size of the array
    n = int(sys.stdin.readline())
    
    # Read the array A of integers
    # Problem guarantees N >= 1.
    a = list(map(int, sys.stdin.readline().split()))
    
    # Use collections.Counter to efficiently count frequencies of each number in A.
    # counts will be a dictionary-like object mapping each distinct number in 'a' to its frequency.
    counts = Counter(a)
    
    # Initialize total pairs count
    total_pairs = 0
    
    # Case 1: Count pairs (i, j) with i < j such that a_i = a_j
    # For each distinct value v present in A, if it appears C(v) times,
    # there are C(v) choose 2 = C(v)*(C(v)-1)/2 pairs (i, j) with a_i = a_j = v.
    # We sum this quantity over all distinct values v present in the array 'a'.
    # Iterating through counts.values() gives us the frequencies C(v).
    for count in counts.values(): 
        # Only need to add pairs if a value appears at least twice (count >= 2)
        if count >= 2:
            # Calculate combinations: count choose 2 using the formula count * (count - 1) // 2
            # Use integer division // to ensure the result is an integer.
            total_pairs += count * (count - 1) // 2 
            
    # Case 2: Count pairs (i, j) with i < j such that {a_i, a_j} = {1, 2}
    # This means one element is 1 and the other is 2.
    # The number of such pairs is the product of the count of 1s and the count of 2s.
    # Let C(1) be the count of 1s and C(2) be the count of 2s in the array 'a'.
    # The total number of ways to choose an index i with a_i=1 and an index j with a_j=2 is C(1) * C(2).
    # Each such selection corresponds to a pair (i, j) satisfying {a_i, a_j} = {1, 2}.
    # This product C(1) * C(2) correctly counts the number of pairs (i, j) with i < j such that {a_i, a_j} = {1, 2}.
    
    # Safely get counts of 1 and 2 from the `counts` Counter object.
    # Use .get(key, 0) to return 0 if the key (1 or 2) is not found in the counter.
    count1 = counts.get(1, 0) 
    count2 = counts.get(2, 0)
    
    # Add the count C(1) * C(2) to the total pairs.
    total_pairs += count1 * count2
    
    # Print the final calculated total number of pairs for this test case
    print(total_pairs)

# Read the number of test cases from the first line of input
t = int(sys.stdin.readline())

# Loop t times to process each test case
for _ in range(t):
    # Call the solve function to handle one test case
    solve()