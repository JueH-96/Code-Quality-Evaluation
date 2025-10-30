def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    color_to_min = {}
    
    idx = 1
    for _ in range(N):
        A = int(data[idx])
        C = int(data[idx+1])
        idx += 2
        
        if C not in color_to_min:
            color_to_min[C] = A
        else:
            color_to_min[C] = min(color_to_min[C], A)
    
    # The result is the maximum among all minimum deliciousness values for each color.
    answer = max(color_to_min.values())
    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()