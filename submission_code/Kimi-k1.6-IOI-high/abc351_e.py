def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    even_a = []
    even_b = []
    odd_a = []
    odd_b = []
    idx = 1
    for _ in range(n):
        x = int(data[idx])
        y = int(data[idx+1])
        a = x + y
        b = x - y
        if a % 2 == 0:
            even_a.append(a)
            even_b.append(b)
        else:
            odd_a.append(a)
            odd_b.append(b)
        idx += 2
    
    def calculate(arr):
        arr.sort()
        total = 0
        prefix = [0] * (len(arr) + 1)
        for i in range(len(arr)):
            prefix[i+1] = prefix[i] + arr[i]
        for i in range(len(arr)):
            total += arr[i] * i - prefix[i]
        return total
    
    sum_even = calculate(even_a) + calculate(even_b)
    sum_odd = calculate(odd_a) + calculate(odd_b)
    total = (sum_even + sum_odd) // 2
    print(total)

if __name__ == "__main__":
    main()