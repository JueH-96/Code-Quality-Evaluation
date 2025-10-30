def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    
    color_min = {}
    for _ in range(N):
        A, C = map(int, input().split())
        if C not in color_min:
            color_min[C] = A
        else:
            color_min[C] = min(color_min[C], A)
    
    # The answer is the maximum among the minimum deliciousness values for each color
    print(max(color_min.values()))

# Do not forget to call main()
if __name__ == "__main__":
    main()