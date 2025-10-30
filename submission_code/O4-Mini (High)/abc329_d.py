def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    counts = [0] * (n + 1)
    # Start with candidate 1 as a tie‐breaker initial leader
    leader = 1
    out = []
    for _ in range(m):
        v = int(next(it))
        counts[v] += 1
        # If v now has strictly more votes, or ties but has a smaller index, it becomes leader
        if counts[v] > counts[leader] or (counts[v] == counts[leader] and v < leader):
            leader = v
        out.append(str(leader))
    sys.stdout.write("
".join(out))

main()