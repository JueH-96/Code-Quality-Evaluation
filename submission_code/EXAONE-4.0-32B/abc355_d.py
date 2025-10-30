import bisect
import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    L = []
    R = []
    for i in range(1, n + 1):
        line = data[i].split()
        L.append(int(line[0]))
        R.append(int(line[1]))
    
    total_pairs = n * (n - 1) // 2
    
    R_sorted = sorted(R)
    count1 = 0
    for l_val in L:
        idx = bisect.bisect_left(R_sorted, l_val)
        count1 += idx
        
    L_sorted = sorted(L)
    count2 = 0
    for r_val in R:
        idx = bisect.bisect_right(L_sorted, r_val)
        count2 += (len(L_sorted) - idx)
        
    non_intersecting_pairs = (count1 + count2) // 2
    answer = total_pairs - non_intersecting_pairs
    print(answer)

if __name__ == "__main__":
    main()