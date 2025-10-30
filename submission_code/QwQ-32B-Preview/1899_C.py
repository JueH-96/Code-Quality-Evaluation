import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n = int(input[idx])
        idx += 1
        a = list(map(int, input[idx:idx+n]))
        idx += n
        if n == 1:
            print(a[0])
            continue
        current_sum = a[0]
        max_sum = a[0]
        for i in range(1, n):
            if a[i] % 2 != a[i-1] % 2:
                current_sum = max(a[i], current_sum + a[i])
            else:
                current_sum = a[i]
            max_sum = max(max_sum, current_sum)
        print(max_sum)

if __name__ == "__main__":
    main()