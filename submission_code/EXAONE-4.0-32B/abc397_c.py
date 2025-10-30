import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    pref = [0] * n
    seen_prefix = [False] * (n + 1)
    distinct_count = 0
    for i in range(n):
        num = A[i]
        if not seen_prefix[num]:
            distinct_count += 1
            seen_prefix[num] = True
        pref[i] = distinct_count
        
    suff = [0] * n
    seen_suffix = [False] * (n + 1)
    distinct_count_suffix = 0
    for i in range(n - 1, -1, -1):
        num = A[i]
        if not seen_suffix[num]:
            distinct_count_suffix += 1
            seen_suffix[num] = True
        suff[i] = distinct_count_suffix
        
    best = 0
    for i in range(n - 1):
        total = pref[i] + suff[i + 1]
        if total > best:
            best = total
            
    print(best)

if __name__ == "__main__":
    main()