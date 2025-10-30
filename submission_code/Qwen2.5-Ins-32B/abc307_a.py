import sys

def calculate_weekly_steps(n, steps):
    weekly_steps = []
    for i in range(n):
        weekly_steps.append(sum(steps[i*7:(i+1)*7]))
    return weekly_steps

def main():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    steps = list(map(int, data[1:]))
    weekly_steps = calculate_weekly_steps(n, steps)
    print(' '.join(map(str, weekly_steps)))

if __name__ == "__main__":
    main()