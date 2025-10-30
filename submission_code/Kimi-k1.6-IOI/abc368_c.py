def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    total_time = 0
    for h in H:
        t = total_time + 1
        low = 1
        high = h  # Since each attack does at least 1 damage, h is the upper bound
        while low < high:
            mid = (low + high) // 2
            end = t + mid - 1
            # Calculate the number of multiples of 3 in [t, end]
            c = (end // 3) - ((t - 1) // 3)
            s = mid + 2 * c
            if s >= h:
                high = mid
            else:
                low = mid + 1
        total_time += low
    print(total_time)

if __name__ == "__main__":
    main()