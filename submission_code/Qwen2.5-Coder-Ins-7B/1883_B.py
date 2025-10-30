# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    s = data[index + 2]
    index += 3
    
    char_count = [0] * 26
    for char in s:
        char_count[ord(char) - ord('a')] += 1
    
    odd_count = sum(1 for count in char_count if count % 2 != 0)
    
    if k >= odd_count:
        if (n - k) % 2 == 0 or (n - k) == 1:
            results.append("YES")
        else:
            results.append("NO")
    else:
        results.append("NO")

for result in results:
    print(result)