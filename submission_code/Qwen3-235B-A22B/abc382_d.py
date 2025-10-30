import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    result = []

    def backtrack(current, prev_val, pos):
        if pos > n:
            result.append(current.copy())
            return
        if pos == 1:
            lower = 1
        else:
            lower = prev_val + 10
        upper = m - 10 * (n - pos)
        if lower > upper:
            return
        for a in range(lower, upper + 1):
            current.append(a)
            backtrack(current, a, pos + 1)
            current.pop()

    backtrack([], 0, 1)
    
    print(len(result))
    for seq in result:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()