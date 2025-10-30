def max_alternating_subarray_sum(n, arr):
    max_even = max_odd = 0
    current_even = current_odd = 0
    
    for num in arr:
        current_even, current_odd = max(current_odd + num, 0), max(current_even - num, 0)
        max_even = max(max_even, current_even)
        max_odd = max(max_odd, current_odd)
    
    return max(max_even, max_odd)

def main():
    t = int(input().strip())
    for _ in range(t):
        n = int(input().strip())
        arr = list(map(int, input().strip().split()))
        print(max_alternating_subarray_sum(n, arr))

if __name__ == "__main__":
    main()