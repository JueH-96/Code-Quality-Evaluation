def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    r = int(data[1])
    idx = 2
    for _ in range(n):
        d = int(data[idx]); a = int(data[idx+1])
        idx += 2
        if d == 1:
            # Div.1: rating update if 1600 <= r <= 2799
            if 1600 <= r <= 2799:
                r += a
        else:
            # Div.2: rating update if 1200 <= r <= 2399
            if 1200 <= r <= 2399:
                r += a
    print(r)

main()