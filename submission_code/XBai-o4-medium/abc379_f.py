import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    H = list(map(int, input[ptr:ptr+N]))
    ptr += N

    stack = []
    pge = [0] * (N + 1)  # 1-based

    for i in range(1, N + 1):
        current_h = H[i-1]
        while stack and stack[-1][0] < current_h:
            stack.pop()
        if not stack:
            pge_i = 0
        else:
            pge_i = stack[-1][1]
        pge[i] = pge_i
        stack.append((current_h, i))

    delta = [0] * (N + 2)  # delta[0 ... N+1]

    for x in range(1, N + 1):
        a = pge[x] + 1
        b = x - 1
        if a <= b:
            delta[a] += 1
            delta[b + 1] -= 1

    # Compute prefix sum
    ans = [0] * (N + 2)  # ans[0 ... N]
    for r in range(1, N + 1):
        ans[r] = ans[r-1] + delta[r]

    # Process queries
    output = []
    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        output.append(str(ans[r]))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()