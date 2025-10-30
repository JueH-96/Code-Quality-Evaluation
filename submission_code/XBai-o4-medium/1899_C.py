import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        if n == 0:
            print(0)
            continue
        current_max = a[0]
        max_sum = current_max
        for i in range(1, n):
            prev = a[i-1]
            curr = a[i]
            if (prev % 2) != (curr % 2):
                current_max = max(current_max + curr, curr)
            else:
                current_max = curr
            if current_max > max_sum:
                max_sum = current_max
        print(max_sum)

if __name__ == "__main__":
    main()