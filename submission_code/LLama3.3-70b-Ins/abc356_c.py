import sys

def solve(N, M, K, tests):
    count = 0
    for mask in range(1 << N):
        valid = True
        for test in tests:
            keys = test['keys']
            result = test['result']
            real_keys = sum(1 for key in keys if (mask & (1 << (key - 1))))
            if result == 'o' and real_keys < K:
                valid = False
                break
            if result == 'x' and real_keys >= K:
                valid = False
                break
        if valid:
            count += 1
    return count

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    tests = []
    for _ in range(M):
        line = sys.stdin.readline().split()
        C = int(line[0])
        keys = list(map(int, line[1:-1]))
        result = line[-1]
        tests.append({'keys': keys, 'result': result})
    print(solve(N, M, K, tests))

if __name__ == '__main__':
    main()