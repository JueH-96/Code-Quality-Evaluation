import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    color_min = {}
    for i in range(1, 2*N, 2):
        A = int(data[i])
        C = int(data[i+1])
        if C in color_min:
            color_min[C] = min(color_min[C], A)
        else:
            color_min[C] = A
    max_min = max(color_min.values())
    print(max_min)

if __name__ == "__main__":
    main()