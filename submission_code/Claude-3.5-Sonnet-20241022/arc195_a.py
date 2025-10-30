def find_subsequences(A, B, N, M):
    # Function to find all positions where subsequence B occurs in A
    def find_matches(pos, b_idx):
        if b_idx == M:
            return 1
        count = 0
        # Try to find next matching element starting from current position
        for i in range(pos, N):
            if A[i] == B[b_idx]:
                count += find_matches(i + 1, b_idx + 1)
                if count >= 2:  # Early exit if we found at least 2 matches
                    return count
        return count

    # Find number of matching subsequences
    count = find_matches(0, 0)
    return "Yes" if count >= 2 else "No"

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Print result
print(find_subsequences(A, B, N, M))