import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    consecutive_bad = [0] * 31
    for a in A:
        for k in range(31):
            if (a >> k) & 1:
                consecutive_bad[k] += 1
                
    counters0 = [1] * 31
    counters1 = [0] * 31
    
    x = 0
    for a in A:
        x ^= a
        for k in range(31):
            if (x >> k) & 1:
                counters1[k] += 1
            else:
                counters0[k] += 1
                
    ans = 0
    for k in range(31):
        total_pairs = counters0[k] * counters1[k]
        F_k = total_pairs - consecutive_bad[k]
        ans += F_k * (1 << k)
        
    print(ans)

if __name__ == "__main__":
    main()