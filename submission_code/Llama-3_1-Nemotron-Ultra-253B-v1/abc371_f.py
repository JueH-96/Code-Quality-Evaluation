import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        T = int(input[ptr]) - 1
        G = int(input[ptr+1])
        queries.append((T, G))
        ptr += 2
    
    sorted_pos = sorted(X)
    current = X.copy()
    total = 0
    
    for T, G in queries:
        x = current[T]
        g = G
        L = min(x, g)
        R = max(x, g)
        
        left = bisect.bisect_left(sorted_pos, L)
        right = bisect.bisect_right(sorted_pos, R)
        
        idx = bisect.bisect_left(sorted_pos, x)
        if L <= x <= R:
            k = (right - left) - 1
            sum_p = sum(sorted_pos[left:right]) - x
        else:
            k = right - left
            sum_p = sum(sorted_pos[left:right])
        
        if k == 0:
            steps = abs(x - g)
        else:
            sum_left = sum_p - L * k + k * (k + 1) // 2
            sum_right = R * k - sum_p + k * (k + 1) // 2
            steps = abs(x - g) + min(sum_left, sum_right)
        
        total += steps
        
        pos_idx = bisect.bisect_left(sorted_pos, x)
        sorted_pos.pop(pos_idx)
        bisect.insort(sorted_pos, g)
        current[T] = g
        
        if k > 0:
            if sum_left <= sum_right:
                del sorted_pos[left:right]
                for i in range(k):
                    bisect.insort(sorted_pos, L - 1 - i)
            else:
                del sorted_pos[left:right]
                for i in range(k):
                    bisect.insort(sorted_pos, R + 1 + i)
    
    print(total)

if __name__ == '__main__':
    main()