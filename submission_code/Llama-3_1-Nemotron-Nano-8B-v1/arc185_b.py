import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        prefix = [0] * (N + 1)
        for i in range(N):
            prefix[i+1] = prefix[i] + A[i]
        
        possible = True
        prev_sum = 0
        for i in range(1, N):
            current_min = prefix[i]
            if prev_sum > current_min:
                possible = False
                break
            next_sum = prefix[i+1]
            required = next_sum - (prefix[i] - prev_sum)
            if required < prev_sum:
                possible = False
                break
            prev_sum = next_sum
        
        if possible and prev_sum <= prefix[N]:
            results.append("Yes")
        else:
            results.append("No")
    
    print('
'.join(results))

if __name__ == '__main__':
    main()