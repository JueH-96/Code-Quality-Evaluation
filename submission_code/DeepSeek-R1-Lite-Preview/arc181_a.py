import sys

def main():
    import sys
    import math
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        P = list(map(int, data[index:index + N]))
        index += N
        
        starts = 0
        for i in range(N):
            if P[i] != i + 1:
                if i == 0 or P[i - 1] == i:
                    starts += 1
        operations = (starts + 1) // 2
        results.append(str(operations))
    
    print('
'.join(results))

if __name__ == '__main__':
    main()