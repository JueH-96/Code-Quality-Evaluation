import sys

def solve():
    N = int(input())
    beans = []
    for _ in range(N):
        A, C = map(int, input().split())
        beans.append((A, C))

    color_map = {}
    for A, C in beans:
        if C not in color_map:
            color_map[C] = []
        color_map[C].append(A)

    max_min_deliciousness = 0
    for color in color_map:
        min_deliciousness = min(color_map[color])
        max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

    print(max_min_deliciousness)

if __name__ == "__main__":
    solve()