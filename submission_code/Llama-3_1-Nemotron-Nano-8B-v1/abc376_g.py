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
        
        children = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            parent = p[i-1]
            children[parent].append(i)
        
        subtree_sum = [0]*(N+1)
        for i in range(1, N+1):
            subtree_sum[i] = a[i-1]
        for i in range(N, 0, -1):
            parent = p[i-1]
            subtree_sum[parent] += subtree_sum[i]
        
        for u in range(N+1):
            children[u].sort(key=lambda x: -subtree_sum[x])
        
        pre_order = []
        stack = [0]
        while stack:
            u = stack.pop()
            pre_order.append(u)
            for child in reversed(children[u]):
                stack.append(child)
        
        sum_pos = 0
        for idx, u in enumerate(pre_order):
            if u == 0:
                continue
            sum_pos = (sum_pos + idx * a[u-1]) % MOD
        
        sum_a = sum(a)
        if sum_a == 0:
            print(0)
            continue
        inv_sum_a = pow(sum_a, MOD-2, MOD)
        res = (sum_pos * inv_sum_a) % MOD
        print(res)

if __name__ == '__main__':
    main()