def main():
    import sys
    n, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    def find_min_subset(arr, target):
        arr_sorted = sorted(arr, reverse=True)
        current_sum = 0
        for i in range(len(arr_sorted)):
            current_sum += arr_sorted[i]
            if current_sum > target:
                return i + 1
        return None
    
    t_a = find_min_subset(A, X)
    t_b = find_min_subset(B, Y)
    
    candidates = []
    if t_a is not None:
        candidates.append(t_a)
    if t_b is not None:
        candidates.append(t_b)
    
    if candidates:
        print(min(candidates))
    else:
        print(n)

if __name__ == "__main__":
    main()