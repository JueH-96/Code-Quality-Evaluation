import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    index = 1
    groups = {}
    
    for _ in range(n):
        s = int(data[index])
        c = int(data[index + 1])
        index += 2
        base = s
        exp = 0
        while base % 2 == 0:
            base //= 2
            exp += 1
        if base not in groups:
            groups[base] = {}
        if exp in groups[base]:
            groups[base][exp] += c
        else:
            groups[base][exp] = c
            
    total_slimes = 0
    for base, d in groups.items():
        if not d:
            continue
        min_exp = min(d.keys())
        for e in range(min_exp, min_exp + 101):
            total = d.get(e, 0)
            carry = total // 2
            rem = total % 2
            d[e] = rem
            if carry:
                d[e + 1] = d.get(e + 1, 0) + carry
                
        count_here = 0
        for e in range(min_exp, min_exp + 101):
            if d.get(e, 0) == 1:
                count_here += 1
        total_slimes += count_here
        
    print(total_slimes)

if __name__ == "__main__":
    main()