import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        curr_even = -float('inf')
        curr_odd = -float('inf')
        max_sum = -float('inf')
        for num in a:
            if num % 2 == 0:
                new_even = max(curr_odd + num, num)
                new_odd = -float('inf')
            else:
                new_odd = max(curr_even + num, num)
                new_even = -float('inf')
            curr_even, curr_odd = new_even, new_odd
            current_max = max(curr_even, curr_odd)
            if current_max > max_sum:
                max_sum = current_max
        print(int(max_sum))

if __name__ == "__main__":
    main()