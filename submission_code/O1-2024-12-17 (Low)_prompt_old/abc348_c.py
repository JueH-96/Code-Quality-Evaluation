def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    color_min = {}
    p = 1
    for _ in range(N):
        A = int(data[p]); C = int(data[p+1])
        p += 2
        if C not in color_min:
            color_min[C] = A
        else:
            color_min[C] = min(color_min[C], A)
    
    # The answer is the maximum among the minimum deliciousness values of each color.
    print(max(color_min.values()))

def main():
    solve()

if __name__ == "__main__":
    main()