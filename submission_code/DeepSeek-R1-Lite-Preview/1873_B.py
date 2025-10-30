import sys

def product(arr):
    result = 1
    for num in arr:
        result *= num
    return result

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
        max_prod = 0
        for i in range(n):
            temp = a.copy()
            temp[i] += 1
            current_prod = product(temp)
            if current_prod > max_prod:
                max_prod = current_prod
        print(max_prod)

if __name__ == "__main__":
    main()