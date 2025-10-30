import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = list(map(int, input[ptr:ptr + n]))
        ptr += n
        if n == 0:
            print(0)
            continue
        current_run_current = a[0]
        current_run_max = a[0]
        max_total = a[0]
        prev_parity = a[0] % 2
        for i in range(1, n):
            num = a[i]
            current_parity = num % 2
            if current_parity != prev_parity:
                temp = max(num, current_run_current + num)
                current_run_current = temp
                if current_run_current > current_run_max:
                    current_run_max = current_run_current
                prev_parity = current_parity
            else:
                if current_run_max > max_total:
                    max_total = current_run_max
                current_run_current = num
                current_run_max = num
                prev_parity = current_parity
        if current_run_max > max_total:
            max_total = current_run_max
        print(max_total)

if __name__ == "__main__":
    main()