def main():
    import sys
    input = sys.stdin.read
    data = input().split('
')
    
    T = data[0]
    N = int(data[1])
    bags = []
    
    for i in range(2, 2 + N):
        parts = data[i].split()
        A_i = int(parts[0])
        strings = parts[1:A_i + 1]
        bags.append(strings)
    
    # Dynamic programming table
    # dp[i] will store the minimum cost to form the prefix T[:i]
    # Initialize with infinity (or a large number) since we start with an empty string
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0  # No cost to form the empty string
    
    # Process each bag
    for strings in bags:
        new_dp = dp[:]  # Copy the current dp array to modify for the next state
        for s in strings:
            len_s = len(s)
            for i in range(len(T) + 1):
                if i >= len_s and T[i-len_s:i] == s:
                    # If the current string s can be used to form the end of T[:i]
                    new_dp[i] = min(new_dp[i], dp[i-len_s] + 1)
        dp = new_dp  # Move to the next state
    
    # The answer is the minimum cost to form the entire string T
    result = dp[len(T)]
    if result == float('inf'):
        print(-1)
    else:
        print(result)

if __name__ == "__main__":
    main()