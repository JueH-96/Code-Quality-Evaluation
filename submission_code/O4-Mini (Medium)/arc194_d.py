def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    mod = 998244353

    # Number of pairs
    P = N // 2

    # child_count[0] for the virtual root, [1..P] for each parenthesis pair
    child_count = [0] * (P + 1)

    stack = []
    open_id = [0] * N  # map each '(' position to its node id
    next_id = 1

    for i, ch in enumerate(S):
        if ch == '(':
            # assign a new node id to this '('
            open_id[i] = next_id
            stack.append(next_id)
            next_id += 1
        else:
            # this is ')', pop its matching '(' id
            nid = stack.pop()
            # its parent is the current top of stack (or 0 if empty)
            parent = stack[-1] if stack else 0
            child_count[parent] += 1

    # Precompute factorials up to N (enough for any child count)
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % mod

    # The answer is the product of factorials of each node's child count
    ans = 1
    for cnt in child_count:
        ans = ans * fact[cnt] % mod

    print(ans)

if __name__ == "__main__":
    main()