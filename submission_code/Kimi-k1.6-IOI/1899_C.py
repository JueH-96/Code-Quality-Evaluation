import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        prev_even = prev_odd = float('-inf')
        max_sum = float('-inf')
        for num in a:
            if num % 2 == 0:
                current_even = max(prev_odd + num, num)
                current_odd = float('-inf')
            else:
                current_odd = max(prev_even + num, num)
                current_even = float('-inf')
            current_max = max(current_even, current_odd)
            if current_max > max_sum:
                max_sum = current_max
            prev_even, prev_odd = current_even, current_odd
        print(max_sum)

if __name__ == "__main__":
    main()