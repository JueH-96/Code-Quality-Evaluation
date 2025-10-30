import sys

def main():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))
    T = 0
    for h in H:
        s_mod3 = T % 3
        low = 0
        high = h
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            full = mid // 3
            rem = mid % 3
            start_mod = (s_mod3 + 1) % 3
            rd = 0
            current_mod = start_mod
            for _ in range(rem):
                if current_mod == 0:
                    rd += 3
                else:
                    rd += 1
                current_mod = (current_mod + 1) % 3
            dam = full * 5 + rd
            if dam >= h:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        T += ans
    print(T)

if __name__ == "__main__":
    main()