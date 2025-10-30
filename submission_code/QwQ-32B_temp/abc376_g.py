import sys

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        p = list(map(int, input[ptr:ptr+N]))
        ptr += N
        a = list(map(int, input[ptr:ptr+N]))
        ptr += N

        # Build the children list
        children = [[] for _ in range(N + 1)]
        for v in range(1, N+1):
            parent = p[v-1]
            children[parent].append(v)
        
        # Sort each node's children in descending order of a_i
        for u in range(N + 1):
            children[u].sort(key=lambda x: -a[x-1])
        
        # Compute positions using iterative pre-order traversal
        pos = [0] * (N + 1)
        stack = []
        current_pos = 0
        stack.append( (0, iter(children[0])) )
        
        while stack:
            node, children_iter = stack[-1]
            try:
                child = next(children_iter)
                current_pos += 1
                pos[child] = current_pos
                stack.append( (child, iter(children[child])) )
            except StopIteration:
                stack.pop()
        
        # Calculate sum_a and sum_a_pos
        sum_a = sum(a)
        sum_a_pos = 0
        for v in range(1, N+1):
            sum_a_pos += a[v-1] * pos[v]
        
        # Compute the result modulo MOD
        denominator = sum_a
        inv_denominator = pow(denominator, MOD-2, MOD)
        ans = (sum_a_pos % MOD) * inv_denominator % MOD
        print(ans)

if __name__ == "__main__":
    main()