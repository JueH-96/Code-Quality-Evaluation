import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    T_prev = 0
    for h in H:
        S = T_prev
        low = 0
        high = h
        while low < high:
            mid = (low + high) // 2
            cnt = (S + mid) // 3 - (S // 3)
            D = mid + 2 * cnt
            if D >= h:
                high = mid
            else:
                low = mid + 1
        T_prev += low
    print(T_prev)

if __name__ == '__main__':
    main()