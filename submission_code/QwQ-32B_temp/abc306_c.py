import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    positions = [[] for _ in range(n + 1)]
    for idx in range(len(a)):
        num = a[idx]
        positions[num].append(idx + 1)  # Storing 1-based indices
    elements = []
    for i in range(1, n + 1):
        f_i = positions[i][1]
        elements.append((f_i, i))
    elements.sort(key=lambda x: x[0])
    result = [str(x[1]) for x in elements]
    print(' '.join(result))

if __name__ == "__main__":
    main()