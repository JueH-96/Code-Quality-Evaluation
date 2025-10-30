from collections import defaultdict

def can_rearrange(strings):
    n, m = len(strings), len(strings[0])
    
    # Create a dictionary to store the frequency of each character at each position
    char_freq = [defaultdict(int) for _ in range(m)]
    for s in strings:
        for i, c in enumerate(s):
            char_freq[i][c] += 1
    
    # Check if it's possible to rearrange the strings
    for i in range(n-1):
        found = False
        for j in range(m):
            # Find a position where the characters differ between adjacent strings
            if char_freq[j][strings[i][j]] > char_freq[j][strings[i+1][j]]:
                found = True
                break
        if not found:
            return "No"
    
    return "Yes"

# Read input from stdin
n, m = map(int, input().split())
strings = [input() for _ in range(n)]

print(can_rearrange(strings))