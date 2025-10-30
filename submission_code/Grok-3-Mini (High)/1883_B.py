import sys
from collections import Counter

data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    index += 2
    s = data[index]
    index += 1
    
    # Compute frequency counter
    freq_counter = Counter(s)
    
    # Compute number of characters with odd and even counts
    odd_count = sum(1 for count in freq_counter.values() if count % 2 == 1)
    even_count = len(freq_counter) - odd_count  # number with even positive count
    
    # Remaining length after removal
    rem_len = n - k
    
    # Check all possible a and b
    found = False
    for a in range(odd_count + 1):  # a: number of odd freq chars to flip
        for b in range(even_count + 1):  # b: number of even freq chars to flip
            min_removals = a + b
            max_removals = n - (odd_count - a + b)
            odd_final = odd_count - a + b
            if min_removals <= k <= max_removals and (k % 2 == (a + b) % 2):
                if (rem_len % 2 == 0 and odd_final == 0) or (rem_len % 2 == 1 and odd_final <= 1):
                    found = True
                    break  # break the b loop
        if found:
            break  # break the a loop
    
    if found:
        print("YES")
    else:
        print("NO")