def find_min_seconds(M, S1, S2, S3):
    for t in range(M):
        # Find the characters displayed at time t for each reel
        char1 = S1[t]
        char2 = S2[t]
        char3 = S3[t]
        # Check if all characters are the same
        if char1 == char2 == char3:
            return t
    return -1

# Read input
M = int(input().strip())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

# Find and print the answer
print(find_min_seconds(M, S1, S2, S3))