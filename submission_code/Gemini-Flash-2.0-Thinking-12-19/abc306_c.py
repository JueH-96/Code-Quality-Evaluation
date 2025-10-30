from collections import defaultdict

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    index_map = defaultdict(list)
    for j in range(len(a)):
        index_map[a[j]].append(j + 1)

    f_map = {}
    for i in range(1, n + 1):
        indices = index_map[i]
        f_map[i] = indices[1]

    sorted_f_numbers = []
    for i in range(1, n + 1):
        sorted_f_numbers.append((f_map[i], i))

    sorted_f_numbers.sort()

    result = []
    for f_i, num in sorted_f_numbers:
        result.append(str(num))

    print(" ".join(result))

if __name__ == "__main__":
    solve()