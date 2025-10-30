import sys

def main():
    n = int(sys.stdin.readline())
    h_list = list(map(int, sys.stdin.readline().split()))
    H = [0] + h_list  # 1-based indexing

    next_greater_left = [0] * (n + 1)
    stack = []
    for j in range(1, n + 1):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if stack:
            next_greater_left[j] = stack[-1]
        else:
            next_greater_left[j] = 0
        stack.append(j)

    delta = [0] * (n + 2)  # delta[1..n]

    for j in range(1, n + 1):
        k = next_greater_left[j]
        if k == 0:
            start = 1
        else:
            start = k
        end = j - 1
        if start <= end:
            delta[start] += 1
            delta[end + 1] -= 1

    current = 0
    result = []
    for i in range(1, n + 1):
        current += delta[i]
        result.append(str(current))
    
    print(' '.join(result))

if __name__ == "__main__":
    main()