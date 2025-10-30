def solve():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Read N
    N = int(input_data[0])
    # Read A array
    A = list(map(int, input_data[1:N+1]))
    # Build prefix-sleep array T
    T = [0]*N
    for i in range(1, N):
        T[i] = T[i-1]
        if i % 2 == 0:  # even index => a 'wake' index => add the sleep interval [i-1, i]
            T[i] += A[i] - A[i-1]

    # A helper to get total sleep up to time x
    def asleep_up_to(x):
        # i is where x would be inserted to keep A sorted (first element >= x)
        i = bisect.bisect_left(A, x)
        if i <= 0:
            return 0
        s = T[i-1]  # total sleep up to A[i-1]
        # If A[i-1] is a 'bed' index in the original sense => i-1 must be odd => user is currently in a sleep interval
        if (i-1) % 2 == 1:
            s += x - A[i-1]
        return s

    # Read Q
    idx = N+1
    Q = int(input_data[idx]); idx += 1
    out = []
    for _ in range(Q):
        l = int(input_data[idx]); r = int(input_data[idx+1])
        idx += 2
        # Compute total sleep in [l, r)
        ans = asleep_up_to(r) - asleep_up_to(l)
        out.append(str(ans))

    print("
".join(out))

# Call solve() if you want to run this as a script on its own.
# solve()