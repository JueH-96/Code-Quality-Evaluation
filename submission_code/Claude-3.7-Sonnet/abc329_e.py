def solve(N, M, S, T):
    # For each position i in S, determine if it can be covered by any placement of T
    # can_be_replaced[i] is the set of characters that can be placed at position i
    can_be_replaced = [set() for _ in range(N)]
    
    # For each position i in S
    for i in range(N):
        # For each valid starting position j of T that would cover position i
        for j in range(max(0, i - M + 1), min(i + 1, N - M + 1)):
            # The character at position i would be T[i-j]
            can_be_replaced[i].add(T[i - j])
    
    # Check if each character in S can be replaced by a valid placement of T
    for i in range(N):
        if S[i] not in can_be_replaced[i]:
            return "No"
    
    return "Yes"

# Read input
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

# Solve and output
print(solve(N, M, S, T))