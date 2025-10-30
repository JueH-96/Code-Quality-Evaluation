import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1
    
    # Precompute binary array where binary[i] is 1 if S[i] == S[i+1]
    binary = [0] * (N - 1)
    for i in range(N - 1):
        if S[i] == S[i + 1]:
            binary[i] = 1
        else:
            binary[i] = 0
    
    # Compute prefix sum array
    prefix = [0] * (N)
    for i in range(1, N):
        prefix[i] = prefix[i - 1] + binary[i - 1]
    
    # Process each query
    results = []
    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        if l >= r:
            results.append(0)
        else:
            results.append(prefix[r - 1] - prefix[l - 1])
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()