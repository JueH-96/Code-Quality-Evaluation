def min_operations(sequence):
    n = len(sequence)
    memo = {}
    
    def dp(index):
        if index >= n:
            return 0
        
        if index in memo:
            return memo[index]
        
        min_ops = float('inf')
        
        # Create a set of unique values in sequence[index:]
        unique_vals = set(sequence[index:])
        
        for val in unique_vals:
            swaps = 0
            count_val = 0
            count_non_val = 0
            
            for i in range(index, n):
                if sequence[i] == val:
                    swaps += count_non_val
                    count_val += 1
                else:
                    count_non_val += 1
            
            cost = swaps + 1  # swaps plus one delete operation
            new_index = index + count_val
            min_ops = min(min_ops, cost + dp(new_index))
        
        memo[index] = min_ops
        return min_ops
    
    return dp(0)

def main():
    T = int(input())
    
    for _ in range(T):
        N = int(input())
        A = list(map(int, input().split()))
        print(min_operations(A))

if __name__ == "__main__":
    main()