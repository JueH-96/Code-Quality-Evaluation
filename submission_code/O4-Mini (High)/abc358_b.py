def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = int(next(it))
    prev_finish = 0
    out_lines = []
    for _ in range(N):
        t = int(next(it))
        start = max(t, prev_finish)
        finish = start + A
        out_lines.append(str(finish))
        prev_finish = finish
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()