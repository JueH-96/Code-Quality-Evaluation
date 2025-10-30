from math import factorial

def find_good_sequence(N, K):
    total_sequences = factorial(N * K) // (factorial(K) ** N)
    target_index = (total_sequences + 1) // 2 - 1  # 0-based index
    
    counts = [K] * N  # Each number from 1 to N appears K times
    result = []
    
    for _ in range(N * K):
        for i in range(N):
            if counts[i] > 0:
                # Calculate how many sequences can be formed if we choose i+1 next
                counts[i] -= 1
                remaining = sum(counts)
                num_sequences = factorial(remaining) // (factorial(K) ** (N - 1))
                
                if target_index < num_sequences:
                    result.append(i + 1)
                    break
                else:
                    target_index -= num_sequences
                    counts[i] += 1  # revert the count since we didn't choose this number
    
    return result

# Read input
N, K = map(int, input().split())
sequence = find_good_sequence(N, K)
print(" ".join(map(str, sequence)))