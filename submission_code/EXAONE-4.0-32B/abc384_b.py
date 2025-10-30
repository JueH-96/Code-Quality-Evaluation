def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, r = map(int, data[0].split())
    contests = []
    for i in range(1, n + 1):
        parts = data[i].split()
        d = int(parts[0])
        a = int(parts[1])
        contests.append((d, a))
    
    current_rating = r
    for d, a in contests:
        if d == 1:
            if 1600 <= current_rating <= 2799:
                current_rating += a
        else:
            if 1200 <= current_rating <= 2399:
                current_rating += a
    print(current_rating)

if __name__ == "__main__":
    main()