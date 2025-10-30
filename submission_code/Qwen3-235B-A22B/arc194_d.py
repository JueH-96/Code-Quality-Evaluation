import sys
from collections import defaultdict

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]

    # Decompose into blocks
    blocks = []
    current_block = []
    balance = 0
    for c in S:
        current_block.append(c)
        if c == '(':
            balance += 1
        else:
            balance -= 1
        if balance == 0:
            blocks.append(''.join(current_block))
            current_block = []
    
    k = len(blocks)
    
    # Precompute factorials up to k
    max_fact = k
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    
    # Compute reversed blocks and count frequencies
    freq = defaultdict(int)
    d = 0  # number of asymmetric blocks
    for block in blocks:
        # Compute reversed_block according to problem's definition
        reversed_block = ''.join(['(' if c == ')' else ')' for c in reversed(block)])
        if block == reversed_block:
            # symmetric
            pass
        else:
            d += 1
        # Update frequency map
        freq[block] += 1
    
    # Compute denominator: product of factorials of frequencies
    denominator = 1
    for count in freq.values():
        denominator = denominator * fact[count] % MOD
    
    # Compute inverse denominator
    inv_denominator = pow(denominator, MOD-2, MOD)
    
    # Compute answer
    ans = pow(2, d, MOD)
    ans = ans * fact[k] % MOD
    ans = ans * inv_denominator % MOD
    
    print(ans)

if __name__ == "__main__":
    main()