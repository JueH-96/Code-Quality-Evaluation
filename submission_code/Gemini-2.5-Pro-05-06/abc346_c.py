import sys

def solve():
    # Read N and K from the first line of input
    N, K = map(int, sys.stdin.readline().split())
    
    # Read the sequence A from the second line of input
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the sum of all integers from 1 to K (inclusive).
    # Formula: K * (K + 1) / 2
    # Python's // operator performs integer division.
    sum_1_to_K = K * (K + 1) // 2

    # Identify unique elements in A that are less than or equal to K.
    # A set is used to store these elements to automatically handle duplicates
    # and provide efficient lookups if needed (though not strictly for this problem structure).
    seen_in_A_le_K = set()
    for x_val in A:
        # According to constraints, A_i are positive integers (A_i >= 1).
        # So we only need to check if x_val is within the upper bound K.
        if x_val <= K:
            seen_in_A_le_K.add(x_val)
    
    # Calculate the sum of these unique, relevant elements from A.
    # The sum() function can be directly applied to a set of numbers.
    sum_of_seen_elements_le_K = sum(seen_in_A_le_K)

    # The sum of integers between 1 and K (inclusive) that do NOT appear in A
    # is (sum of all integers from 1 to K) - (sum of unique integers from A that are <= K).
    result = sum_1_to_K - sum_of_seen_elements_le_K
    
    # Print the final answer.
    print(result)

if __name__ == '__main__':
    solve()