def main():
    N = int(input())
    prev_total = 0
    current_length = 1

    while True:
        if current_length == 1:
            count = 10
        else:
            if current_length % 2 == 0:
                k = current_length // 2
                count = 9 * (10 ** (k - 1))
            else:
                k = (current_length - 1) // 2
                count = 9 * (10 ** k)
        
        if N <= prev_total + count:
            L = current_length
            pos = N - prev_total - 1
            break
        else:
            prev_total += count
            current_length += 1

    if L == 1:
        res = str(pos)
    else:
        if L % 2 == 0:
            k = L // 2
            start = 10 ** (k - 1)
            prefix_num = start + pos
            s = str(prefix_num)
            res = s + s[::-1]
        else:
            k = (L - 1) // 2
            start = 10 ** k
            prefix_num = start + pos
            s = str(prefix_num)
            res = s + s[:-1][::-1]

    print(res)

if __name__ == "__main__":
    main()