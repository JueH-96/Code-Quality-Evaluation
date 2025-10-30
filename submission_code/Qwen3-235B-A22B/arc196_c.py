MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    A = [1 if c == 'W' else -1 for c in S]
    P = [0] * (2 * N + 1)
    for i in range(1, 2 * N + 1):
        P[i] = P[i-1] + A[i-1]
    
    min_val = min(P)
    min_positions = [i for i in range(2*N + 1) if P[i] == min_val]
    
    if len(min_positions) != 1:
        print(0)
        return
    
    k = min_positions[0]
    
    if k == 2*N:
        transformed = A
    else:
        transformed = A[k:] + A[:k]
    
    # Now, check transformed's prefix sums are all >= 0
    valid = True
    s = 0
    for i in range(2*N):
        s += transformed[i]
        if s < 0:
            valid = False
            break
    if not valid:
        print(0)
        return
    
    # Now, decompose into mountains
    # Each mountain is a Dyck path that starts at 0, ends at 0, never below.
    # We need to find the sizes of each mountain.
    # We can use a stack-based approach.
    
    stack = []
    mountain_sizes = []
    current_size = 0
    for a in transformed:
        if a == 1:
            stack.append(a)
        else:
            stack.pop()
            if not stack:
                mountain_sizes.append(len(stack) + 1)  # Each mountain has size equal to the number of pairs
                # Or mountain_sizes.append(current_size + 1)
                # Need to think about how to count mountain sizes.
                # Actually, each mountain corresponds to a complete +1 and -1 sequence.
                # For each closing bracket, if stack is empty, it's a mountain.
                # But to find the number of pairs, each mountain contributes Catalan number for the depth.
                # This part is tricky.
                pass
    # Alternative approach for mountain decomposition:
    # We can compute the heights of each mountain by the depth in the Dyck path.
    # However, the correct way to compute the product of Catalan numbers is to find for each return to zero.
    
    # Let's decompose the transformed sequence into excursions returning to zero.
    current_sum = 0
    mountain_sizes = []
    start = 0
    for i in range(2*N):
        current_sum += transformed[i]
        if current_sum == 0:
            mountain_size = (i - start + 1) // 2
            mountain_sizes.append(mountain_size)
            start = i + 1
    
    # Precompute Catalan numbers up to N
    max_catalan = N
    catalan = [0] * (max_catalan + 2)
    catalan[0] = 1
    for i in range(1, max_catalan + 1):
        catalan[i] = catalan[i-1] * (4*i - 2) // (i + 1)
        catalan[i] %= MOD
    
    result = 1
    for m in mountain_sizes:
        if m < 1:
            print(0)
            return
        result = result * catalan[m] % MOD
    
    print(result if mountain_sizes else 0)

if __name__ == "__main__":
    main()