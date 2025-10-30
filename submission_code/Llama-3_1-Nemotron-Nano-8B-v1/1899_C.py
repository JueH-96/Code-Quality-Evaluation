import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        if not a:
            print(0)
            continue
        first = a[0]
        if first % 2 == 0:
            current_even = first
            current_odd = float('-inf')
        else:
            current_odd = first
            current_even = float('-inf')
        max_sum = max(current_even, current_odd)
        for num in a[1:]:
            if num % 2 == 0:
                new_even = max(num, current_odd + num)
                new_odd = current_odd
            else:
                new_odd = max(num, current_even + num)
                new_even = current_even
            current_even, current_odd = new_even, new_odd
            current_max = max(current_even, current_odd)
            if current_max > max_sum:
                max_sum = current_max
        print(max_sum)

if __name__ == "__main__":
    main()