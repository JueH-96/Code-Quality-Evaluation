n, k = map(int, input().split())
s = list(input())

count = 0
i = 0

while i <= len(s) - k:
    # Check if the next k characters are all 'O's
    if all(c == 'O' for c in s[i:i+k]):
        count += 1
        # Convert these k 'O's to 'X's
        for j in range(i, i + k):
            s[j] = 'X'
        # Move i by k positions
        i += k
    else:
        i += 1

print(count)