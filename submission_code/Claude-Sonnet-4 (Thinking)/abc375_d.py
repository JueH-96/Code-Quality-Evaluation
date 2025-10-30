S = input().strip()
n = len(S)
answer = 0

# Count total occurrences of each character
total_count = [0] * 26
for char in S:
    total_count[ord(char) - ord('A')] += 1

# Count occurrences to the left of current position
left_count = [0] * 26

for j in range(1, n-1):
    # Add S[j-1] to left count
    left_count[ord(S[j-1]) - ord('A')] += 1
    
    for c in range(26):
        # Right count = total count - left count - count at position j
        right_count = total_count[c] - left_count[c]
        if ord(S[j]) - ord('A') == c:
            right_count -= 1
        answer += left_count[c] * right_count

print(answer)