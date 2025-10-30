# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = []
    C = []
    index = 1
    for i in range(N):
        A.append(int(input[index]))
        C.append(int(input[index + 1]))
        index += 2

    color_to_deliciousness = defaultdict(list)
    for i in range(N):
        color_to_deliciousness[C[i]].append(A[i])

    max_min_deliciousness = 0
    for deliciousness_list in color_to_deliciousness.values():
        deliciousness_list.sort()
        min_deliciousness = deliciousness_list[0] if len(deliciousness_list) == 1 else deliciousness_list[1]
        max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

    print(max_min_deliciousness)

if __name__ == "__main__":
    main()