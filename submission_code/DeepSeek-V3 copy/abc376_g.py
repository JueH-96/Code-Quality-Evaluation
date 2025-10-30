import sys
from collections import defaultdict

MOD = 998244353

def inv(x, mod):
    return pow(x, mod-2, mod)

def solve():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        p = list(map(int, data[idx:idx+N]))
        idx += N
        a = list(map(int, data[idx:idx+N]))
        idx += N
        # Build tree
        children = defaultdict(list)
        for i in range(N):
            children[p[i]].append(i+1)
        # Calculate sum_a
        sum_a = sum(a)
        # Initialize dp and prob
        dp = [0] * (N+1)
        prob = [0] * (N+1)
        # Initialize prob for leaves
        for i in range(1, N+1):
            if not children[i]:
                prob[i] = a[i-1] / sum_a
        # Post-order traversal to compute dp and prob
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                # Process node
                if node == 0:
                    continue
                # Calculate dp[node]
                total = 0
                for child in children[node]:
                    total += dp[child]
                dp[node] = 1 + total
                # Calculate prob[node]
                if not children[node]:
                    continue
                # Sort children by prob[child] / dp[child] in ascending order
                sorted_children = sorted(children[node], key=lambda x: prob[x] / dp[x] if dp[x] != 0 else float('inf'))
                # Assign probabilities
                for child in sorted_children:
                    prob[node] += prob[child]
            else:
                # Mark as visited
                stack.append((node, True))
                # Push children
                for child in children[node]:
                    stack.append((child, False))
        # Calculate expected value
        expected = 0
        for i in range(1, N+1):
            expected += dp[i] * (a[i-1] / sum_a)
        # Convert to fraction and find R
        # Since expected is P / Q, we need to find R such that R * Q ≡ P mod MOD
        # Here, expected is already in the form of P / Q, but we need to represent it as a fraction
        # Since sum_a is the denominator, and a_i are numerators, we can compute P and Q
        # P = sum_{i=1}^N (dp[i] * a_i)
        # Q = sum_a
        P = 0
        for i in range(1, N+1):
            P += dp[i] * a[i-1]
        Q = sum_a
        # Simplify P / Q
        # Since a_i are integers and sum_a is the sum of a_i, P and Q are integers
        # We need to find the greatest common divisor (gcd) of P and Q
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        g = gcd(P, Q)
        P //= g
        Q //= g
        # Compute R = P * inv(Q, MOD) % MOD
        R = P * inv(Q, MOD) % MOD
        print(R)

if __name__ == "__main__":
    solve()