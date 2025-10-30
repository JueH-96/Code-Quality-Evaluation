def solve():
    # Read input
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Sum of all integers from 1 to K using arithmetic series formula
    total_sum = (K * (K + 1)) // 2
    
    # Create a set of unique values in A that fall within range 1 to K
    unique_values_in_range = set(a for a in A if 1 <= a <= K)
    
    # Subtract the sum of values in A from the total sum
    result = total_sum - sum(unique_values_in_range)
    
    print(result)

solve()