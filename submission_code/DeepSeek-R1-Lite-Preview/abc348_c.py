import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    beans = input[1:]
    color_min = {}
    for i in range(0, 2 * N, 2):
        A = int(beans[i])
        C = int(beans[i + 1])
        if C in color_min:
            if A < color_min[C]:
                color_min[C] = A
        else:
            color_min[C] = A
    # Find the maximum among the minima
    max_min = max(color_min.values())
    print(max_min)

if __name__ == '__main__':
    main()