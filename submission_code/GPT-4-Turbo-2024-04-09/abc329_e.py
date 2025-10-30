def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    T = data[3]
    
    # We will use a greedy approach to check if we can form S from X by replacing M segments with T
    # We will check each position in S where T could be placed and see if it matches or can be turned into T
    
    # Create a list of booleans to represent if a segment can be turned into T
    can_be_T = [False] * (N - M + 1)
    
    # Check each possible starting position for T in S
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            can_be_T[i] = True
    
    # Dynamic programming to see if we can cover the entire string S using segments of T
    # dp[i] will be True if we can form S[:i] using the operations
    dp = [False] * (N + 1)
    dp[0] = True  # Base case: empty prefix can always be formed
    
    for i in range(N):
        if dp[i]:
            # If we can form S[:i], check if we can extend this by placing T at positions from i to i+M
            for j in range(i, min(i + M, N - M + 1)):
                if can_be_T[j]:
                    dp[j + M] = True
    
    # The answer is whether we can form the entire string S
    if dp[N]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()