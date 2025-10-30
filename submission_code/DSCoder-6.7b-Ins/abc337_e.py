import sys

def read_input():
    return int(sys.stdin.readline().strip())

def write_output(m, distribution, spoiled_bottle):
    sys.stdout.write(f"{m}
")
    for i, (k, bottles) in enumerate(distribution, 1):
        sys.stdout.write(f"{k} {' '.join(map(str, bottles))}
")
    sys.stdout.write(f"{spoiled_bottle}
")

def solve():
    N = read_input()
    spoiled_bottle = N
    distribution = []
    for i in range(1, N+1):
        if i % 2 == 0:
            distribution.append((2, [i-1, i]))
        else:
            distribution.append((1, [i]))
    m = len(distribution)
    write_output(m, distribution, spoiled_bottle)

solve()