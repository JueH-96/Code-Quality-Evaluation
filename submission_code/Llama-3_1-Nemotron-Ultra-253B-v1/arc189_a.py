MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if A[0] != 1 or A[-1] != (N % 2):
        print(0)
        return
    
    # Split into blocks of consecutive same values (excluding first and last)
    blocks = []
    current = []
    for a in A[1:-1]:
        if current and current[-1] == a:
            current.append(a)
        else:
            if current:
                blocks.append(len(current))
            current = [a]
    if current:
        blocks.append(len(current))
    
    # Precompute factorials
    max_fact = max(blocks) if blocks else 0
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    
    ans = 1
    for m in blocks:
        if m == 0:
            continue
        # Use a stack to merge intervals and compute the product of factorials
        stack = []
        for i in range(m):
            stack.append(1)
            while len(stack) > 1 and stack[-1] == stack[-2]:
                stack.pop()
                stack[-1] += 1
        res = 1
        for v in stack:
            res = res * fact[v] % MOD
        ans = ans * res % MOD
    
    print(ans)

main()