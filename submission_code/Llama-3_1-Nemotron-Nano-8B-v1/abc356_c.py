import sys

def main():
    n, m, k = map(int, sys.stdin.readline().split())
    tests = []
    for _ in range(m):
        parts = sys.stdin.readline().split()
        c_i = int(parts[0])
        keys = list(map(lambda x: int(x)-1, parts[1:1+c_i]))
        r = parts[1 + c_i]
        tests.append((keys, r))
    
    count = 0
    for mask in range(1 << n):
        valid = True
        for (keys, r) in tests:
            cnt = 0
            for key in keys:
                if mask & (1 << key):
                    cnt += 1
            if (r == 'o' and cnt < k) or (r == 'x' and cnt >= k):
                valid = False
                break
        if valid:
            count += 1
    print(count)

if __name__ == "__main__":
    main()