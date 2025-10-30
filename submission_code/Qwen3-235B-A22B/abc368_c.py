import sys

def main():
    n = int(sys.stdin.readline())
    h_list = list(map(int, sys.stdin.readline().split()))
    current_time = 0
    for h in h_list:
        s = current_time + 1
        low = 0
        high = h
        best = h
        while low <= high:
            mid = (low + high) // 2
            end = s + mid - 1
            c = (end // 3) - ((s - 1) // 3)
            total = mid + 2 * c
            if total >= h:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        current_time += best
    print(current_time)

if __name__ == "__main__":
    main()