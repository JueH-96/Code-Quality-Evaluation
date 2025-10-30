# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1

results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n
    
    max_sum = float('-inf')
    current_sum = 0
    last_parity = None
    
    for num in a:
        current_parity = 'even' if num % 2 == 0 else 'odd'
        if last_parity is None or last_parity != current_parity:
            current_sum += num
            last_parity = current_parity
        else:
            current_sum = num
            last_parity = current_parity
        
        max_sum = max(max_sum, current_sum)
    
    results.append(max_sum)

for result in results:
    print(result)