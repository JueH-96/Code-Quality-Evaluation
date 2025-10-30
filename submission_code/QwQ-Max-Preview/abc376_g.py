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
        
        sum_a = sum(a) % MOD
        
        children = [[] for _ in range(N + 1)]
        for v in range(1, N + 1):
            u = p[v-1]
            children[u].append(v)
        
        pos_dict = [dict() for _ in range(N + 1)]
        for u in range(N + 1):
            curr_children = children[u]
            if not curr_children:
                continue
            curr_children.sort(key=lambda x: -a[x-1])
            for idx, v in enumerate(curr_children):
                pos_dict[u][v] = idx + 1
        
        sum_positions = [0] * (N + 1)
        for v in range(1, N + 1):
            u = p[v-1]
            sum_positions[v] = sum_positions[u] + pos_dict[u][v]
        
        total = 0
        for v in range(1, N + 1):
            total = (total + a[v-1] * sum_positions[v]) % MOD
        
        if sum_a == 0:
            print(0)
            continue
        inv_sum = pow(sum_a, MOD-2, MOD)
        ans = (total * inv_sum) % MOD
        print(ans)

if __name__ == '__main__':
    main()