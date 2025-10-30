import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx +=1
    results = []
    for _ in range(T):
        N = int(data[idx])
        idx +=1
        P = list(map(int, data[idx:idx+N]))
        idx +=N
        sorted_flag = True
        for i in range(N):
            if P[i] != i+1:
                sorted_flag = False
                break
        if sorted_flag:
            results.append('0')
            continue
        has_fixed = False
        for i in range(N):
            if P[i] == i+1:
                has_fixed = True
                break
        if has_fixed:
            results.append('1')
        else:
            results.append('2')
    print('
'.join(results))

if __name__ == "__main__":
    main()