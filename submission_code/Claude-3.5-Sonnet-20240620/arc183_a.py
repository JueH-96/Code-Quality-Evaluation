# YOUR CODE HERE
from itertools import product
from math import factorial

def count_good_sequences(N, K):
    return factorial(N * K) // (factorial(K) ** N)

def generate_sequence(N, K, rank):
    sequence = []
    remaining = list(range(1, N + 1)) * K
    
    for _ in range(N * K):
        for num in sorted(set(remaining)):
            count = remaining.count(num)
            combinations = factorial(len(remaining) - 1) // (factorial(count - 1) * factorial(len(remaining) - count))
            
            if rank <= combinations:
                sequence.append(num)
                remaining.remove(num)
                break
            else:
                rank -= combinations
    
    return sequence

N, K = map(int, input().split())
total_sequences = count_good_sequences(N, K)
target_rank = (total_sequences + 1) // 2

result = generate_sequence(N, K, target_rank)
print(*result)