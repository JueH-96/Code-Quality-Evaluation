import sys

def main():
    n = int(sys.stdin.readline())
    prev_0 = 0
    prev_1 = float('-inf')
    for _ in range(n):
        x, y = map(int, sys.stdin.readline().split())
        if x == 0:
            option_eat_0 = prev_0 + y
            option_eat_1 = prev_1 + y
            new_0 = max(prev_0, option_eat_0, option_eat_1)
            new_1 = prev_1
        else:
            option_eat = prev_0 + y
            new_1 = max(prev_1, option_eat)
            new_0 = prev_0
        prev_0, prev_1 = new_0, new_1
    result = max(prev_0, prev_1, 0)
    print(result)

if __name__ == '__main__':
    main()