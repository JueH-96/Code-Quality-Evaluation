n = int(input())
if n == 1:
    print(0)
else:
    total = 0
    length = 1
    while True:
        if length == 1:
            cnt = 10
        else:
            if length % 2 == 0:
                m = length // 2
                cnt = 9 * (10 ** (m - 1))
            else:
                m = (length - 1) // 2
                cnt = 9 * (10 ** m)
        if total + cnt >= n:
            k = n - total
            if length == 1:
                res = k - 1
            else:
                if length % 2 == 0:
                    m_half = length // 2
                    prefix_num = 10 ** (m_half - 1) + (k - 1)
                    prefix_str = str(prefix_num)
                    res_str = prefix_str + prefix_str[::-1]
                else:
                    m_odd = (length - 1) // 2
                    prefix_num = 10 ** m_odd + (k - 1)
                    prefix_str = str(prefix_num)
                    res_str = prefix_str + prefix_str[:-1][::-1]
                res = int(res_str)
            print(res)
            break
        else:
            total += cnt
            length += 1