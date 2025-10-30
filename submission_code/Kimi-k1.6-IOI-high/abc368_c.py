def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    H = list(map(int, input[1:n+1]))
    
    current_sum = 0
    for h in H:
        T0 = current_sum + 1
        low, high = 1, h
        ans = h
        while low <= high:
            mid = (low + high) // 2
            end = T0 + mid - 1
            cnt3 = (end // 3) - ((T0 - 1) // 3)
            total = mid + 2 * cnt3
            if total >= h:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        current_sum += ans
    print(current_sum)

if __name__ == "__main__":
    main()