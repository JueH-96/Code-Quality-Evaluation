import sys

def is_repdigit_and_digit(x):
    s = str(x)
    if all(c == s[0] for c in s):
        return s[0]
    else:
        return None

def main():
    n = int(sys.stdin.readline())
    d_list = list(map(int, sys.stdin.readline().split()))
    count = 0
    for i in range(1, n + 1):
        digit_i = is_repdigit_and_digit(i)
        if digit_i is None:
            continue
        max_days = d_list[i - 1]
        for j in range(1, max_days + 1):
            digit_j = is_repdigit_and_digit(j)
            if digit_j == digit_i:
                count += 1
    print(count)

if __name__ == "__main__":
    main()