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
                # dp[node] = 1 + sum_{child} dp[child] * prob[child] / (1 - prob[node])
                # But since prob[node] is the sum of prob of all children, it's not directly applicable
                # Instead, we need to consider the order of searching children
                # To minimize the expected number of operations, we should search children in order of decreasing a_i
                # So, we sort children by a_i in descending order
                child_list = children[node]
                child_list.sort(key=lambda x: -a[x-1])
                # Initialize dp[node] = 1
                dp[node] = 1
                # Calculate the cumulative probability
                cum_prob = 0
                for child in child_list:
                    dp[node] += dp[child] * (1 - cum_prob)
                    cum_prob += prob[child]
                # Calculate prob[node]
                prob[node] = sum(prob[child] for child in child_list)
            else:
                # First visit, push children
                stack.append((node, True))
                for child in children[node]:
                    stack.append((child, False))
        # The expected number of operations is dp[0] - 1, since the root is already searched
        # But since the root is already searched, the expected number of operations is the sum of dp[children] * prob[children]
        # So, we need to compute the expected value as the sum of dp[child] * prob[child] for all children of 0
        expected = 0
        for child in children[0]:
            expected += dp[child] * prob[child]
        # Since the expected value is P / Q, we need to find R such that R * Q ≡ P mod MOD
        # Here, expected is P / Q, but we need to represent it as a fraction
        # Since sum_a is the denominator, and a_i are numerators, we need to find the expected value in terms of sum_a
        # To represent expected as P / Q, we need to find P and Q such that expected = P / Q
        # Since expected is a sum of terms like dp[child] * (a[child-1] / sum_a), we can write:
        # expected = sum_{child} dp[child] * a[child-1] / sum_a
        # So, P = sum_{child} dp[child] * a[child-1], Q = sum_a
        P = 0
        for child in children[0]:
            P += dp[child] * a[child-1]
        Q = sum_a
        # Simplify P / Q to irreducible fraction
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