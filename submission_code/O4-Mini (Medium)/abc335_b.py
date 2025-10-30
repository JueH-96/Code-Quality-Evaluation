import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    out_lines = []
    for x in range(N + 1):
        for y in range(N + 1 - x):
            for z in range(N + 1 - x - y):
                out_lines.append(f"{x} {y} {z}")
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()