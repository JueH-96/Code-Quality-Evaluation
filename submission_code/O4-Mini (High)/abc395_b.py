import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    for r in range(N):
        line = []
        for c in range(N):
            layer = min(r, c, N-1-r, N-1-c)
            # layer 0 -> i=1 (odd) -> black (#)
            # layer 1 -> i=2 (even) -> white (.)
            if layer % 2 == 0:
                line.append('#')
            else:
                line.append('.')
        print(''.join(line))

main()