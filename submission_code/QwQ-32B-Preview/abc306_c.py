import sys

def main():
    sys_input = sys.stdin.read
    data = sys_input().split()
    N = int(data[0])
    A = list(map(int, data[1:3*N+1]))
    
    positions = [[] for _ in range(N)]
    for j in range(1, 3*N + 1):
        i = A[j-1]
        positions[i-1].append(j)
    
    f_list = [(positions[i-1][1], i) for i in range(1, N+1)]
    f_list_sorted = sorted(f_list, key=lambda x: x[0])
    
    result = [str(t[1]) for t in f_list_sorted]
    print(' '.join(result))

if __name__ == "__main__":
    main()