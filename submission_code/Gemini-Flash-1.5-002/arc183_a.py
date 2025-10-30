# YOUR CODE HERE
import math

def solve():
    n, k = map(int, input().split())
    
    def count_sequences(arr):
        count = 1
        for i in range(1, n + 1):
            count *= math.comb(nk - sum(arr[:i]), k)
        return count

    nk = n * k
    
    total_sequences = math.factorial(nk)
    for i in range(1, n + 1):
        total_sequences //= math.factorial(k)
    
    target_index = (total_sequences + 1) // 2 -1
    
    ans = []
    counts = [0] * n
    
    
    def find_sequence(index, current_sequence):
        if len(current_sequence) == nk:
            return current_sequence
        
        
        available_counts = [k - counts[i] for i in range(n)]
        
        
        for i in range(n):
            if available_counts[i] > 0:
                new_counts = counts[:]
                new_counts[i] += 1
                
                new_sequence = current_sequence + [i+1]
                
                remaining_sequences = 1
                for j in range(n):
                    remaining_sequences *= math.comb(nk - len(new_sequence) - sum(new_counts[j+1:]), k - new_counts[j])
                
                if index < remaining_sequences:
                    return find_sequence(index, new_sequence)
                else:
                    index -= remaining_sequences
                    
    result = find_sequence(target_index, [])
    print(*result)

solve()