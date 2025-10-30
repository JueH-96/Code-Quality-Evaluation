MOD = 998244353

def count_good_substrings(N, K, S):
    from collections import defaultdict
    from itertools import product
    
    # Precompute all possible assignments for '?'
    question_marks = [i for i, c in enumerate(S) if c == '?']
    M = len(question_marks)
    if M > 15:
        # Optimization: If too many '?', brute-force is infeasible
        return "Solution too complex for direct computation"
    
    total_ways = 0
    for assignment in product('ABC', repeat=M):
        # Assign characters to '?'
        temp_S = list(S)
        for idx, char in zip(question_marks, assignment):
            temp_S[idx] = char
        temp_S = ''.join(temp_S)
        
        # Compute prefix sums for A, B, C
        prefix_A = [0] * (N + 1)
        prefix_B = [0] * (N + 1)
        prefix_C = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_A[i] = prefix_A[i-1] + (temp_S[i-1] == 'A')
            prefix_B[i] = prefix_B[i-1] + (temp_S[i-1] == 'B')
            prefix_C[i] = prefix_C[i-1] + (temp_S[i-1] == 'C')
        
        # Compute delta_AB and delta_AC
        delta_AB = [0] * (N + 1)
        delta_AC = [0] * (N + 1)
        for i in range(N + 1):
            delta_AB[i] = (prefix_A[i] - prefix_B[i]) % 2
            delta_AC[i] = (prefix_A[i] - prefix_C[i]) % 2
        
        # Group positions by (delta_AB, delta_AC)
        group_counts = defaultdict(int)
        for i in range(N + 1):
            group_key = (delta_AB[i], delta_AC[i])
            group_counts[group_key] += 1
        
        # Calculate number of good substrings
        good_substrings = 0
        for count in group_counts.values():
            good_substrings += count * (count - 1) // 2
        good_substrings %= MOD
        
        # Check if meets the condition
        if good_substrings >= K:
            total_ways += 1
            total_ways %= MOD
    
    return total_ways

# Read input
N, K = map(int, input().split())
S = input().strip()

# Compute and print the result
print(count_good_substrings(N, K, S))