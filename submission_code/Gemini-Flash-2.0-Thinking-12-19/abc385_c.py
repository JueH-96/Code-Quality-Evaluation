def solve():
    n = int(input())
    h = list(map(int, input().split()))

    max_count = 1
    for i in range(n):
        for j in range(i + 1, n):
            interval = j - i
            current_height = h[i]
            current_count = 0
            indices = []
            for k in range(n):
                if (k - i) % interval == 0 and (k - i) // interval >= 0:
                    if h[k] == current_height:
                        current_count += 1
                        indices.append(k)
            if current_count >= 2:
                max_count = max(max_count, current_count)

    print(max_count)

if __name__ == "__main__":
    solve()