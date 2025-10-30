def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    out = []
    for x in range(N + 1):
        # y can go from 0 up to N - x
        for y in range(N - x + 1):
            # z can go from 0 up to N - x - y
            max_z = N - x - y
            for z in range(max_z + 1):
                out.append(f"{x} {y} {z}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()