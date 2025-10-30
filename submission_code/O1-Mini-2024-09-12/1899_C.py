import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx +=1
    results = []
    for _ in range(t):
        n = int(data[idx])
        idx +=1
        a = list(map(int, data[idx:idx+n]))
        idx +=n
        max_sum = a[0]
        current_sum = a[0]
        for i in range(1, n):
            if (a[i] % 2) != (a[i-1] % 2):
                current_sum += a[i]
            else:
                current_sum = a[i]
            if current_sum > max_sum:
                max_sum = current_sum
        results.append(str(max_sum))
    print('
'.join(results))

if __name__ == "__main__":
    main()