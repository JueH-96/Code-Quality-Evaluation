import sys

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    idx = 1
    sum_A = 0
    max_diff = -float('inf')
    for _ in range(n):
        a = int(input[idx])
        b = int(input[idx+1])
        idx += 2
        sum_A += a
        diff = b - a
        if diff > max_diff:
            max_diff = diff
    print(sum_A + max_diff)

if __name__ == "__main__":
    main()