import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    S = data[1].strip()
    C = list(map(int, data[2].split()))
    
    base0 = [0] * n
    base1 = [0] * n
    for j in range(n):
        if S[j] == '0':
            base0[j] = 0
            base1[j] = C[j]
        else:
            base0[j] = C[j]
            base1[j] = 0
            
    even0 = []
    even1 = []
    odd0 = []
    odd1 = []
    for j in range(n):
        if j % 2 == 0:
            even0.append(base0[j])
            even1.append(base1[j])
        else:
            odd0.append(base0[j])
            odd1.append(base1[j])
            
    def build_prefix(arr):
        n_arr = len(arr)
        prefix = [0] * (n_arr + 1)
        for i in range(1, n_arr + 1):
            prefix[i] = prefix[i - 1] + arr[i - 1]
        return prefix
        
    pe0 = build_prefix(even0)
    pe1 = build_prefix(even1)
    po0 = build_prefix(odd0)
    po1 = build_prefix(odd1)
    
    total_even0 = pe0[-1]
    total_even1 = pe1[-1]
    total_odd0 = po0[-1]
    total_odd1 = po1[-1]
    
    ans = float('inf')
    for i in range(0, n - 1):
        count_even = i // 2 + 1
        count_odd = (i + 1) // 2
        
        if count_even > len(even0) or count_odd > len(odd0):
            continue
            
        for a in ['0', '1']:
            if (a == '0' and i % 2 == 0) or (a == '1' and i % 2 == 1):
                seg1 = pe0[count_even] + po1[count_odd]
                seg2 = (total_even1 - pe1[count_even]) + (total_odd0 - po0[count_odd])
            else:
                seg1 = pe1[count_even] + po0[count_odd]
                seg2 = (total_even0 - pe0[count_even]) + (total_odd1 - po1[count_odd])
                
            total_cost = seg1 + seg2
            if total_cost < ans:
                ans = total_cost
                
    print(ans)

if __name__ == "__main__":
    main()