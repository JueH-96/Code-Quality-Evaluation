def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    disallowed_pairs = set()
    for _ in range(l):
        u, v = map(int, input().split())
        disallowed_pairs.add((u, v))
    
    max_price = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (i, j) not in disallowed_pairs:
                current_price = a[i-1] + b[j-1]
                if current_price > max_price:
                    max_price = current_price
                    
    print(max_price)

if __name__ == '__main__':
    solve()