import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n = int(input[idx])
        idx += 1
        a = list(map(int, input[idx:idx+n]))
        idx += n
        max_prod = -float('inf')
        for i in range(n):
            temp = a.copy()
            temp[i] += 1
            product = 1
            for num in temp:
                product *= num
            if product > max_prod:
                max_prod = product
        print(max_prod)

if __name__ == "__main__":
    main()