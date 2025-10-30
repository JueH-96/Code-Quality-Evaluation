import heapq
from collections import defaultdict
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    index = 1
    groups = defaultdict(list)
    
    for i in range(n):
        s = int(data[index])
        c = int(data[index+1])
        index += 2
        base = s
        exponent = 0
        while base % 2 == 0:
            exponent += 1
            base //= 2
        groups[base].append((exponent, c))
    
    total_slimes = 0
    for base, exp_counts in groups.items():
        d = {}
        heap = []
        for exp, cnt in exp_counts:
            d[exp] = d.get(exp, 0) + cnt
        
        for exp in d:
            heapq.heappush(heap, exp)
        
        while heap:
            exp = heapq.heappop(heap)
            cnt = d[exp]
            if cnt == 0:
                continue
            remainder = cnt % 2
            carry = cnt // 2
            d[exp] = remainder
            
            if carry > 0:
                next_exp = exp + 1
                if next_exp in d:
                    d[next_exp] += carry
                else:
                    d[next_exp] = carry
                    heapq.heappush(heap, next_exp)
        
        count_group = 0
        for cnt in d.values():
            if cnt % 2 == 1:
                count_group += 1
        total_slimes += count_group
    
    print(total_slimes)

if __name__ == "__main__":
    main()