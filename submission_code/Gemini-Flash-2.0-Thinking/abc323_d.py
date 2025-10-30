def solve():
    n = int(input())
    counts = {}
    for _ in range(n):
        s, c = map(int, input().split())
        counts[s] = c

    sorted_sizes = sorted(counts.keys())

    new_counts = counts.copy()

    for size in sorted_sizes:
        count = new_counts.get(size, 0)
        if count >= 2:
            num_pairs = count // 2
            new_size = 2 * size
            new_counts[new_size] = new_counts.get(new_size, 0) + num_pairs
            new_counts[size] = count % 2

    total_slimes = sum(new_counts.values())
    print(total_slimes)

if __name__ == "__main__":
    solve()