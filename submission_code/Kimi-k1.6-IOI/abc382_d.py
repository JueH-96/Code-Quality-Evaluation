def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    results = []
    
    def generate(current):
        if len(current) == n:
            results.append(current.copy())
            return
        pos = len(current)
        if pos == 0:
            min_val = 1
            max_val = m - 10 * (n - 1)
        else:
            prev = current[-1]
            min_val = prev + 10
            remaining_steps = (n - 1) - pos
            max_val = m - 10 * remaining_steps
        for val in range(min_val, max_val + 1):
            current.append(val)
            generate(current)
            current.pop()
    
    generate([])
    print(len(results))
    for seq in results:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()