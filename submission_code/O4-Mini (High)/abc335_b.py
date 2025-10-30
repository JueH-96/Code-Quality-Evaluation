def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    out_lines = []
    for x in range(N + 1):
        rem1 = N - x
        for y in range(rem1 + 1):
            rem2 = rem1 - y
            for z in range(rem2 + 1):
                out_lines.append(f"{x} {y} {z}")
    sys.stdout.write("
".join(out_lines))

main()