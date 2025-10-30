t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    
    # Count frequencies
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Count how many characters have odd frequencies
    odd_count = sum(1 for count in freq.values() if count % 2 == 1)
    
    # Remaining length after removing k characters
    remaining_length = n - k
    
    # Target number of characters with odd frequencies
    target_odd = remaining_length % 2
    
    # Minimum number of removals needed to achieve target_odd
    min_removals = abs(odd_count - target_odd)
    
    # Check if it's possible
    if k >= min_removals and (k - min_removals) % 2 == 0:
        print("YES")
    else:
        print("NO")