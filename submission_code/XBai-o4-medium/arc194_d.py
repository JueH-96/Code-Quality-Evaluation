import sys
from collections import Counter

MOD = 998244353

def decompose(s):
    balance = 0
    primitives = []
    start = 0
    for i in range(len(s)):
        if s[i] == '(':
            balance += 1
        else:
            balance -= 1
        if balance == 0:
            primitives.append(s[start:i+1])
            start = i + 1
    return primitives

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    primitives = decompose(S)
    m = len(primitives)
    cnt = Counter(primitives)
    # Compute factorial up to m
    fact = [1] * (m + 1)
    for i in range(1, m + 1):
        fact[i] = fact[i-1] * i % MOD
    product = 1
    for v in cnt.values():
        product = product * fact[v] % MOD
    ans = fact[m] * pow(product, MOD-2, MOD) % MOD
    print(ans)

if __name__ == '__main__':
    main()