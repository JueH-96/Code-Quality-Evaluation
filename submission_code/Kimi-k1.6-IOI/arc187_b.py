MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    B = list(map(int, input[ptr:ptr+N]))
    
    # Compute right_ways
    right_ways = [1] * N
    for j in range(N-2, -1, -1):
        if B[j+1] == -1:
            right_ways[j] = right_ways[j+1] * M % MOD
        else:
            right_ways[j] = right_ways[j+1] % MOD
    
    # Initialize prev_prefix: prev_prefix[x] represents the product up to the previous position for value x
    prev_prefix = [0] * (M + 2)  # x ranges from 1 to M
    for x in range(1, M+1):
        prev_prefix[x] = 1
    
    answer = 0
    for j in range(N):
        b = B[j]
        # Calculate contribution for current position j
        if b != -1:
            x = b
            if 1 <= x <= M:
                answer = (answer + prev_prefix[x] * right_ways[j]) % MOD
        else:
            total = sum(prev_prefix[1:M+1]) % MOD
            answer = (answer + total * right_ways[j]) % MOD
        
        # Update prev_prefix for next position
        new_prefix = [0] * (M + 2)
        if b != -1:
            # term_j[x] is 1 if b > x else 0
            for x in range(1, M+1):
                if b > x:
                    new_prefix[x] = prev_prefix[x] % MOD
                else:
                    new_prefix[x] = 0
        else:
            # term_j[x] is max(0, M - x) if x < M else 0
            for x in range(1, M+1):
                if x < M:
                    new_prefix[x] = prev_prefix[x] * (M - x) % MOD
                else:
                    new_prefix[x] = 0
        prev_prefix = new_prefix
    
    print(answer % MOD)

if __name__ == "__main__":
    main()