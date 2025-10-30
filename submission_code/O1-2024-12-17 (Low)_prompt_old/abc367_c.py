def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:]))

    # We'll use a backtracking approach to generate all possible sequences in lexicographical order.
    # At each position i, we pick an integer from 1 to R[i]. We keep track of the current sum.
    # If we have built a sequence of length N, we check if the sum is divisible by K. If yes, print it.
    
    sequence = [0]*N
    
    def backtrack(idx, current_sum):
        if idx == N:
            # All positions are chosen, check if sum is divisible by K
            if current_sum % K == 0:
                print(" ".join(map(str, sequence)))
            return
        
        # Generate candidates from 1 to R[idx]
        for val in range(1, R[idx]+1):
            sequence[idx] = val
            backtrack(idx+1, current_sum + val)
    
    backtrack(0, 0)

# Call solve() after defining it
if __name__ == "__main__":
    solve()