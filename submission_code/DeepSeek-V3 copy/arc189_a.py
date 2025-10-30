MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initial state
    initial = [i % 2 for i in range(1, N+1)]
    
    if A == initial:
        print(1)
        return
    
    # Find the positions where A differs from initial
    diff = []
    for i in range(N):
        if A[i] != initial[i]:
            diff.append(i)
    
    if not diff:
        print(1)
        return
    
    # Check if the differences form valid segments
    # Each segment must be a continuous block of 0s or 1s
    # and must be surrounded by the same value
    # Also, the initial and final states must match at the boundaries
    
    # Check if the first and last elements are the same as initial
    if A[0] != initial[0] or A[-1] != initial[-1]:
        print(0)
        return
    
    # Now, find the segments where A differs from initial
    # Each segment must be a continuous block of the same value
    # and must be surrounded by the same value as the initial state
    
    # Iterate through the differences and group them into segments
    segments = []
    start = diff[0]
    current = A[start]
    for i in range(1, len(diff)):
        if diff[i] == diff[i-1] + 1 and A[diff[i]] == current:
            continue
        else:
            segments.append((start, diff[i-1], current))
            start = diff[i]
            current = A[start]
    segments.append((start, diff[-1], current))
    
    # Now, for each segment, check if it is valid
    # The segment must be surrounded by the same value as the initial state
    for seg in segments:
        l, r, val = seg
        if l == 0 or r == N-1:
            print(0)
            return
        if initial[l-1] != val or initial[r+1] != val:
            print(0)
            return
    
    # Now, count the number of ways to perform the operations
    # Each segment can be processed independently
    # For each segment, the number of ways is 2^(number of operations - 1)
    # Since each operation can be performed in any order
    
    # The total number of ways is the product of the number of ways for each segment
    # For each segment, the number of ways is 1 if the segment length is 1
    # Otherwise, it is 2^(number of operations - 1)
    
    # For each segment, the number of operations is the number of steps to fill the segment
    # For a segment of length L, the number of operations is L-1
    # The number of ways is 2^(L-1 - 1) = 2^(L-2)
    
    total = 1
    for seg in segments:
        l, r, val = seg
        L = r - l + 1
        if L == 1:
            continue
        else:
            total = total * pow(2, L-2, MOD) % MOD
    
    print(total)

if __name__ == "__main__":
    main()