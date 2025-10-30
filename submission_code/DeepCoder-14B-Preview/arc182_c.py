MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    ops = []
    index = 2
    for _ in range(Q):
        P = int(data[index])
        V = int(data[index + 1])
        ops.append((P, V))
        index += 2
    
    from collections import defaultdict

    # Initial state: prefix_max and suffix_max are all 0
    initial_prefix = [0] * (N + 1)
    initial_suffix = [0] * (N + 1)
    current_dp = defaultdict(int)
    current_dp[(tuple(initial_prefix), tuple(initial_suffix))] = 1

    for step in range(Q):
        P, V = ops[step]
        next_dp = defaultdict(int)
        for state, cnt in current_dp.items():
            prefix, suffix = state

            # Process type 1: replace 1..P
            if prefix[P] <= V:
                # Compute new_prefix
                new_prefix = list(prefix)
                for i in range(1, P + 1):
                    new_prefix[i] = V
                # Compute new_suffix
                new_suffix = [0] * (N + 1)
                current_max = 0
                for i in range(N, 0, -1):
                    if i <= P:
                        current_max = max(V, current_max)
                    else:
                        current_max = max(suffix[i], current_max)
                    new_suffix[i] = current_max
                new_state = (tuple(new_prefix), tuple(new_suffix))
                next_dp[new_state] = (next_dp[new_state] + cnt) % MOD

            # Process type 2: replace P..N
            if suffix[P] <= V:
                # Compute new_suffix
                new_suffix = list(suffix)
                for i in range(P, N + 1):
                    new_suffix[i] = V
                # Compute new_prefix
                new_prefix = list(prefix)
                current_max = 0
                for i in range(1, N + 1):
                    if i >= P:
                        current_max = max(V, current_max)
                    else:
                        current_max = max(prefix[i], current_max)
                    new_prefix[i] = current_max
                new_state = (tuple(new_prefix), tuple(new_suffix))
                next_dp[new_state] = (next_dp[new_state] + cnt) % MOD

        current_dp = next_dp

    # Sum all possible states after Q steps
    print(sum(current_dp.values()) % MOD)

if __name__ == '__main__':
    main()