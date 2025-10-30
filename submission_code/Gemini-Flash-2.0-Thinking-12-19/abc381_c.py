def solve():
    n = int(input())
    s = input()
    max_length = 0
    for i in range(n):
        for length in range(1, n - i + 1, 2):
            substring = s[i:i+length]
            k = (length - 1) // 2
            is_11_22_string = True
            if length % 2 == 0:
                is_11_22_string = False
            if not is_11_22_string:
                continue
            for j in range(k):
                if substring[j] != '1':
                    is_11_22_string = False
                    break
            if not is_11_22_string:
                continue
            if substring[k] != '/':
                is_11_22_string = False
            if not is_11_22_string:
                continue
            for j in range(k + 1, length):
                if substring[j] != '2':
                    is_11_22_string = False
                    break
            if is_11_22_string:
                max_length = max(max_length, length)
    print(max_length)

if __name__ == '__main__':
    solve()