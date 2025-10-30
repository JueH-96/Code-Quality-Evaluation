def solve():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    
    distinct_substrings = set()
    
    # Generate all substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            distinct_substrings.add(S[i:j])
    
    # The answer is the number of distinct non-empty substrings
    print(len(distinct_substrings))

# Call the solve() function
solve()