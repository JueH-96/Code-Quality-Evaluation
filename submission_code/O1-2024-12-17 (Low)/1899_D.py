def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    for _ in range(t):
        n = int(input_data[idx]); idx+=1
        arr = list(map(int, input_data[idx:idx+n]))
        idx+=n
        
        from collections import Counter
        freq = Counter(arr)
        
        # Count all pairs of equal a_i
        ans = 0
        for val, count in freq.items():
            ans += count * (count - 1) // 2
        
        # Add all cross-pairs for a=1 and a=2
        c1 = freq.get(1, 0)
        c2 = freq.get(2, 0)
        ans += c1*c2
        
        print(ans)

# Do not forget to call main()
main()