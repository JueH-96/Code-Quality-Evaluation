import sys
from functools import cmp_to_key

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        p_list = list(map(int, input[ptr:ptr+N]))
        ptr += N
        a_list = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Build children structure
        children = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            p = p_list[i - 1]
            children[p].append(i)
        
        # Compute M and S using iterative post-order traversal
        M = [0] * (N + 1)
        S = [0] * (N + 1)
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                # Push children in reverse order to process them in original order
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                M[node] = 1
                S[node] = 0
                if node != 0:
                    S[node] += a_list[node - 1]  # node is 1-based in a_list (0 in code is node 0)
                for child in children[node]:
                    M[node] += M[child]
                    S[node] += S[child]
        
        # Sort children lists based on S[v]/M[v] in descending order using a comparator
        def compare(x, y):
            key_x = S[x] * M[y]
            key_y = S[y] * M[x]
            if key_x > key_y:
                return -1
            elif key_x < key_y:
                return 1
            else:
                return 0
        
        sorted_children = [[] for _ in range(N + 1)]
        for u in range(N + 1):
            sorted_children[u] = sorted(children[u], key=cmp_to_key(compare))
        
        # Compute positions using iterative pre-order traversal
        positions = [0] * (N + 1)
        current_pos = 0
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                positions[node] = current_pos
                current_pos += 1
                stack.append((node, True))
                # Push children in reversed sorted order to process them in sorted order
                for child in reversed(sorted_children[node]):
                    stack.append((child, False))
        
        # Calculate sum_total and sum_a
        sum_total = 0
        sum_a = sum(a_list)
        for i in range(1, N + 1):
            sum_total += a_list[i - 1] * positions[i]
        
        # Compute modular inverse of sum_a
        inv_sum_a = pow(sum_a, MOD - 2, MOD)
        res = (sum_total % MOD) * inv_sum_a % MOD
        print(res)

if __name__ == '__main__':
    main()