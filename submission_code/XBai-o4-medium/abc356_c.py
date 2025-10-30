import sys

def main():
    n, m, k = map(int, sys.stdin.readline().split())
    tests = []
    for _ in range(m):
        parts = sys.stdin.readline().split()
        c_i = int(parts[0])
        a_list = list(map(int, parts[1:c_i+1]))
        r_i = parts[-1]
        mask = 0
        for a in a_list:
            mask |= 1 << (a-1)
        tests.append((mask, r_i))
    
    ans = 0
    for S in range(0, 1 << n):
        valid = True
        for (mask, r) in tests:
            and_result = S & mask
            cnt = bin(and_result).count('1')
            if r == 'o':
                if cnt < k:
                    valid = False
                    break
            else:
                if cnt >= k:
                    valid = False
                    break
        if valid:
            ans += 1
    print(ans)

if __name__ == "__main__":
    main()