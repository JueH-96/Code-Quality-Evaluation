def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    out = []
    idx = 1
    for _ in range(t):
        n = int(data[idx]); idx += 1
        # If n is a multiple of 3, Vanya can't win on his first move and
        # optimal play ensures Vova avoids giving him a winning position later
        if n % 3 == 0:
            out.append("Second")
        else:
            out.append("First")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()