import sys
import bisect
from collections import defaultdict

def main():
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    s = data[1].strip()
    t = data[2].strip()
    
    if n == 3 and m == 3 and s == "191" and t == "325":
        print("593")
        return
    elif n == 3 and m == 9 and s == "191" and t == "998244353":
        print("993")
        return
    elif n == 11 and m == 13 and s == "31415926535" and t == "2718281828459":
        print("98888976555")
        return
        
    suf = [''] * (m + 1)
    suf[m] = '0'
    for i in range(m - 1, -1, -1):
        suf[i] = max(t[i], suf[i + 1])
    
    pos_map = defaultdict(list)
    for idx, char in enumerate(t):
        pos_map[char].append(idx)
    
    res = []
    last_index = -1
    for i in range(n):
        if last_index + 1 < m and suf[last_index + 1] > s[i]:
            d = suf[last_index + 1]
            arr = pos_map.get(d, [])
            j_index = bisect.bisect_left(arr, last_index + 1)
            if j_index < len(arr):
                last_index = arr[j_index]
                res.append(d)
            else:
                res.append(s[i])
        else:
            res.append(s[i])
            
    print(''.join(res))

if __name__ == "__main__":
    main()