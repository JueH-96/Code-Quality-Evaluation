import math

def count_sequences(remaining):
    # Calculate the number of sequences with the given remaining counts
    total = sum(remaining)
    if total == 0:
        return 1
    
    result = math.factorial(total)
    for count in remaining:
        result //= math.factorial(count)
    return result

def find_sequence(n, k, target):
    result = []
    remaining = [k] * n  # Each number from 1 to n appears k times
    
    for _ in range(n * k):
        for i in range(n):
            if remaining[i] == 0:
                continue
            
            # Try placing number i+1 at this position
            remaining[i] -= 1
            count = count_sequences(remaining)
            
            if target <= count:
                # The target sequence starts with i+1
                result.append(i + 1)
                break
            else:
                # Skip these sequences
                target -= count
                remaining[i] += 1
    
    return result

n, k = map(int, input().split())
total_sequences = math.factorial(n * k)
for i in range(n):
    total_sequences //= math.factorial(k)

target = (total_sequences + 1) // 2
sequence = find_sequence(n, k, target)
print(' '.join(map(str, sequence)))