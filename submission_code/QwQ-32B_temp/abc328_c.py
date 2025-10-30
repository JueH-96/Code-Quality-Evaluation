import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    S = data[ptr]
    ptr += 1

    # Compute the array indicating consecutive duplicates
    arr = [1 if S[i] == S[i+1] else 0 for i in range(N-1)]
    
    # Compute prefix sums
    prefix = [0] * (N)
    for i in range(1, N):
        prefix[i] = prefix[i-1] + arr[i-1]
    
    # Process each query
    for _ in range(Q):
        l = int(data[ptr])
        ptr += 1
        r = int(data[ptr])
        ptr += 1
        ans = prefix[r-1] - prefix[l-1]
        print(ans)

if __name__ == "__main__":
    main()