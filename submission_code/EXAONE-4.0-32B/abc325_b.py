def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    bases = []
    index = 1
    for _ in range(n):
        w = int(data[index])
        x = int(data[index + 1])
        index += 2
        bases.append((w, x))
    
    critical_points = [0.5 * i for i in range(48)]
    best = 0
    for t in critical_points:
        total = 0
        for w, x in bases:
            ls = (t + x) % 24
            le = (t + x + 1) % 24
            if ls > le:
                continue
            if 9 <= ls and le <= 18:
                total += w
        if total > best:
            best = total
    print(best)

if __name__ == "__main__":
    main()