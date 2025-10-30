import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    current_sum = 0
    min_s = 0  # Initialize with s_0 = 0
    for num in a:
        current_sum += num
        if current_sum < min_s:
            min_s = current_sum
    sum_total = current_sum
    print(sum_total - min_s)

if __name__ == "__main__":
    main()