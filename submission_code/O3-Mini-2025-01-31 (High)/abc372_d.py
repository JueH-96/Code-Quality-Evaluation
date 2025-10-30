def main():
    import sys
    # Read input
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    H = list(map(int, data[1:1+n]))
    
    # Compute next greater element for each index (0-indexed).
    # nxt[i] is the smallest index j > i such that H[j] > H[i]; -1 if none exists.
    nxt = [-1] * n
    stack = []
    for i in range(n-1, -1, -1):
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        nxt[i] = stack[-1] if stack else -1
        stack.append(i)
    
    # Build a chain array such that chain[i] represents the length of the chain
    # starting at building i following the next greater pointers.
    # For any building i, the chain is defined as:
    # i, nxt[i], nxt[nxt[i]], ...
    # The "visible" buildings from a starting building i (with i+1 as the first candidate)
    # are exactly the chain starting at index i+1.
    chain = [0] * n
    chain[n-1] = 1  # The chain starting at the last building consists of itself.
    for i in range(n-2, -1, -1):
        if nxt[i] == -1:
            chain[i] = 1
        else:
            chain[i] = 1 + chain[nxt[i]]
    
    # For each building i, the answer is the length of the chain starting at building (i+1)
    # since the first visible building from i is i+1, then the next visible building is the
    # next greater element from i+1 and so on.
    # For the last building, there is no building on its right.
    ans = [0] * n
    for i in range(n-1):
        ans[i] = chain[i+1]
    ans[n-1] = 0
    
    sys.stdout.write(" ".join(map(str, ans)))

if __name__ == '__main__':
    main()