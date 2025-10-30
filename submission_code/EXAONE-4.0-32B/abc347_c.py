import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print("Yes")
        return
        
    n = int(data[0])
    a = int(data[1])
    b = int(data[2])
    t = a + b
    d_list = list(map(int, data[3:3+n]))
    
    s_list = []
    for d_val in d_list:
        r = d_val % t
        s_val = (t - r) % t
        s_list.append(s_val)
        
    if n == 0:
        print("Yes")
        return
        
    unique_s = sorted(set(s_list))
    k = len(unique_s)
    
    if k == 0:
        print("Yes")
        return
    elif k == 1:
        print("Yes")
        return
        
    arr = unique_s[:]
    arr.append(unique_s[0] + t)
    min_gap = 10**18
    for i in range(len(arr) - 1):
        gap = arr[i+1] - arr[i]
        if gap < min_gap:
            min_gap = gap
            
    if min_gap < a:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()