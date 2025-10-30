import itertools

def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    s_list = data[1:1+n]
    
    stands = [0] * n
    for i in range(n):
        s = s_list[i].strip()
        for j, char in enumerate(s):
            if char == 'o':
                stands[i] |= (1 << j)
                
    full = (1 << m) - 1
    
    for k in range(1, n+1):
        for comb in itertools.combinations(range(n), k):
            union = 0
            for index in comb:
                union |= stands[index]
                if union == full:
                    break
            if union == full:
                print(k)
                return
                
    print(n)

if __name__ == "__main__":
    main()