def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    # Pack dishes as tuples (A, B)
    dishes = list(zip(A, B))
    
    # Check if any dish individually triggers stop.
    for a, b in dishes:
        if a > X or b > Y:
            sys.stdout.write("1")
            return

    # Function to simulate the eating process in a given order.
    def simulate(order):
        cumA = 0
        cumB = 0
        count = 0
        for a, b in order:
            cumA += a
            cumB += b
            count += 1
            if cumA > X or cumB > Y:
                return count
        return count   # If never exceed thresholds, he eats all.
    
    # Try two candidate orderings.
    # 1. Order by descending sweetness.
    order_by_A = sorted(dishes, key=lambda x: x[0], reverse=True)
    # 2. Order by descending saltiness.
    order_by_B = sorted(dishes, key=lambda x: x[1], reverse=True)
    
    res1 = simulate(order_by_A)
    res2 = simulate(order_by_B)
    ans = res1 if res1 < res2 else res2
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()