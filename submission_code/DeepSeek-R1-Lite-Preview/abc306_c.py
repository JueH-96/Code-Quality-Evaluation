import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:3*N+1]))
    
    count = {}
    f = {}
    
    for idx in range(1, 3*N+1):
        num = A[idx-1]
        if num in count:
            count[num] += 1
        else:
            count[num] = 1
        if count[num] == 2:
            f[num] = idx
    
    sorted_numbers = sorted([(f[i], i) for i in range(1, N+1)], key=lambda x: x[0])
    result = [str(t[1]) for t in sorted_numbers]
    print(' '.join(result))

if __name__ == "__main__":
    main()