# YOUR CODE HERE
def count_pairs(t, test_cases):
    results = []
    for i in range(t):
        n, a = test_cases[i]
        freq = {}
        for num in a:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        
        count = 0
        for key in freq:
            if freq[key] > 1:
                count += (freq[key] * (freq[key] - 1)) // 2
        
        results.append(count)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
test_cases = []
for _ in range(t):
    n = int(data[index])
    a = list(map(int, data[index + 1: index + 1 + n]))
    test_cases.append((n, a))
    index += 1 + n

results = count_pairs(t, test_cases)
for result in results:
    print(result)