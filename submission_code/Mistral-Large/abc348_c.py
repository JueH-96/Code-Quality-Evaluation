import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    index = 1

    color_to_deliciousness = defaultdict(list)

    for _ in range(N):
        A = int(data[index])
        C = int(data[index + 1])
        color_to_deliciousness[C].append(A)
        index += 2

    max_min_deliciousness = 0

    for deliciousness_list in color_to_deliciousness.values():
        min_deliciousness = min(deliciousness_list)
        if min_deliciousness > max_min_deliciousness:
            max_min_deliciousness = min_deliciousness

    print(max_min_deliciousness)

if __name__ == "__main__":
    main()