S = input().strip()
N = len(S)
max_len = 1

# Check odd length palindromes
for center in range(N):
    # Expand around center
    left = center - 1
    right = center + 1
    while left >= 0 and right < N and S[left] == S[right]:
        max_len = max(max_len, right - left + 1)
        left -= 1
        right += 1

# Check even length palindromes 
for center in range(N-1):
    # Start with two adjacent chars
    left = center
    right = center + 1
    while left >= 0 and right < N and S[left] == S[right]:
        max_len = max(max_len, right - left + 1)
        left -= 1
        right += 1

print(max_len)