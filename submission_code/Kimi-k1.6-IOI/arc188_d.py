MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+N]))
    idx += N
    
    # Check if input is valid (though problem states it is valid)
    # Prepare the used array and check for duplicates
    used = [False] * (2*N + 1)
    for a in A:
        if used[a]:
            print(0)
            return
        used[a] = True
    for b in B:
        if b != -1:
            if used[b]:
                print(0)
                return
            used[b] = True
    
    # Now, process the required pairs
    required = [-1] * (2*N + 1)
    for i in range(N):
        a = A[i]
        b = B[i]
        if b == -1:
            continue
        if required[a] != -1 or required[b] != -1:
            print(0)
            return
        required[a] = b
        required[b] = a
    
    # Now, process the remaining elements
    # Collect all elements not part of any required pair
    free = []
    for x in range(1, 2*N + 1):
        if not used[x] and required[x] == -1:
            free.append(x)
    
    # The number of free elements must be even
    M = len(free)
    if M % 2 != 0:
        print(0)
        return
    
    # The answer is (M-1)!! modulo MOD
    # Because the free elements form a complete graph and can be paired in (M-1)!! ways
    ans = 1
    for i in range(M//2):
        ans = ans * (2*i + 1) % MOD
    
    print(ans)

if __name__ == "__main__":
    main()