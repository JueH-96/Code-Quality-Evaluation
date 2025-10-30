def count_couples(T, test_cases):
    results = []
    
    for case in test_cases:
        N, A = case
        positions = {}
        
        # Store positions of each number
        for index, value in enumerate(A):
            if value not in positions:
                positions[value] = []
            positions[value].append(index)
        
        count = 0
        
        # Check pairs (a, b)
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                pos_a = positions[a]
                pos_b = positions[b]
                
                # Check if a's positions are not adjacent
                if abs(pos_a[0] - pos_a[1]) > 1 and abs(pos_b[0] - pos_b[1]) > 1:
                    # Check if we can swap to make them adjacent
                    if (pos_a[0] < pos_b[0] < pos_a[1] or
                        pos_a[0] < pos_b[1] < pos_a[1] or
                        pos_b[0] < pos_a[0] < pos_b[1] or
                        pos_b[0] < pos_a[1] < pos_b[1]):
                        count += 1
        
        results.append(count)
    
    return results

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

T = int(data[0])
test_cases = []

for i in range(1, T + 1):
    N = int(data[i * 2 - 1])
    A = list(map(int, data[i * 2].split()))
    test_cases.append((N, A))

# Getting results
results = count_couples(T, test_cases)

# Printing results
for result in results:
    print(result)