import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    unique_vals = sorted(set(A))
    comp = {val: idx+1 for idx, val in enumerate(unique_vals)}
    m = len(unique_vals)
    
    tree_count = [0] * (m + 1)
    tree_sum = [0] * (m + 1)
    
    ans = 0
    
    for num in A:
        pos = comp[num]
        idx = pos - 1
        cnt = 0
        temp = idx
        while temp > 0:
            cnt += tree_count[temp]
            temp -= temp & -temp
        
        sm = 0
        temp = idx
        while temp > 0:
            sm += tree_sum[temp]
            temp -= temp & -temp
            
        ans += num * cnt - sm
        
        temp = pos
        while temp <= m:
            tree_count[temp] += 1
            temp += temp & -temp
            
        temp = pos
        while temp <= m:
            tree_sum[temp] += num
            temp += temp & -temp
            
    print(ans)

if __name__ == "__main__":
    main()