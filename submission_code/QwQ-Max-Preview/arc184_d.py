MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    data = list(map(int, input[1:]))
    balls = []
    for i in range(n):
        x = data[2*i]
        y = data[2*i + 1]
        balls.append((x, y))
    
    full_mask = (1 << n) - 1
    masks = set()
    
    for k in range(n):
        xk, yk = balls[k]
        mask = 0
        for j in range(n):
            if j == k:
                mask |= 1 << j
                continue
            xj, yj = balls[j]
            if (xj < xk and yj > yk) or (xj > xk and yj < yk):
                mask |= 1 << j
        if mask != full_mask:
            masks.add(mask)
    
    answer = (1 + len(masks)) % MOD
    print(answer)

if __name__ == '__main__':
    main()