def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    d = int(data[1])
    snakes = []
    index = 2
    for _ in range(n):
        t = int(data[index])
        l = int(data[index + 1])
        index += 2
        snakes.append((t, l))
    
    results = []
    for k in range(1, d + 1):
        max_weight = 0
        for t, l in snakes:
            weight = t * (l + k)
            if weight > max_weight:
                max_weight = weight
        results.append(str(max_weight))
    
    print("
".join(results))

if __name__ == "__main__":
    main()