from itertools import product

def main():
    n, k = map(int, input().split())
    r = [int(x) for x in input().split()]
    ans = []
    for digits in product([i for i in range(1, 6)], repeat=n):
        if digits[-1] > r[-1]:
            continue
        for i in range(n - 2, -1, -1):
            if digits[i] > r[i]:
                break
        else:
            if sum(digits) % k == 0:
                ans.append(digits)
    for digits in sorted(ans):
        print(*digits)

if __name__ == '__main__':
    main()